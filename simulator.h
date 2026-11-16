#pragma once

// Header-only C++17. Keep dependency configuration consistent across all .cpp files.
// Dependency configuration belongs to the consuming application.
#include <QtGlobal>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <pocketfft_hdronly.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// 1. Public data structures and API
// -----------------------------------------------------------------------------

namespace gaussian_surface {

struct Surface {
    // Node values: grid[iy * nx + ix], bottom-left first. Bounds are inclusive.
    std::vector<qreal> grid;
    qreal minx = 0;
    qreal maxx = 0;
    qreal miny = 0;
    qreal maxy = 0;
    std::size_t nx = 0;
    std::size_t ny = 0;
};

struct Point {
    qreal x = 0;
    qreal y = 0;
    qreal value = 0;
    // Exact target is always value. Relative weight controls the mean trend only.
    // Nonnegative; zero excludes the point from the trend, not from hard constraints.
    qreal weight = 1;
};

enum class VariogramType {
    Matern32,
    Exponential,
    Gaussian,
    Spherical
};

struct CovarianceParameters {
    VariogramType variogram = VariogramType::Matern32;
    // Total prior standard deviation, including the nugget component.
    double standardDeviation = 1.0;
    // Independent nodal variance, in elevation units squared; [0, deviation^2].
    // Controls remain exact: this is not observation-error variance.
    double nugget = 0.0;
    // Major-axis radius: 95% of the structured sill, or the exact cutoff for Spherical.
    double range = 1.0;
    // Minor-axis radius / major-axis radius, in (0, 1]. One means isotropy.
    double anisotropyRatio = 1.0;
    // Counterclockwise from +X, in degrees.
    double azimuthDegrees = 0.0;
};

struct Limits {
    // Limits count elements, not bytes. See README for memory estimates.
    std::size_t maxEmbeddingNodes = 4 * 1024 * 1024;
    std::size_t maxControlPoints = 2048;
};

struct PreparationInfo {
    std::size_t embeddingNx = 0;
    std::size_t embeddingNy = 0;
    std::size_t controlPoints = 0; // After merging identical constraints.
};

class Simulator {
public:
    // Copies inputs. Throws invalid_argument for invalid data and runtime_error
    // for an infeasible/ill-conditioned model or exceeded computational limits.
    Simulator(const Surface& surface, const std::vector<Point>& points,
              const CovarianceParameters& covariance, const Limits& limits = {});

    // Thread-safe concurrent calls on one prepared Simulator.
    // Repeated seeds reproduce a realization in the same build/environment.
    Surface generate(std::uint64_t seed) const;
    PreparationInfo info() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace gaussian_surface

// -----------------------------------------------------------------------------
// 2. Grid validation and bilinear control constraints
// -----------------------------------------------------------------------------

namespace gaussian_surface::detail {

struct GridGeometry {
    std::size_t nx;
    std::size_t ny;
    double minx;
    double maxx;
    double miny;
    double maxy;
    double dx;
    double dy;
};

struct Observation {
    std::array<std::size_t, 4> nodes{};
    std::array<double, 4> weights{};
    double target = 0.0;
    double relativeWeight = 1.0; // Normalized by the mean after merging duplicates.

    double evaluate(const std::vector<double>& values) const;
};

GridGeometry validateSurface(const Surface& surface);
std::vector<Observation> makeObservations(const GridGeometry& geometry,
                                         const std::vector<Point>& points,
                                         std::size_t maxPoints);

struct CellCoordinate {
    std::size_t index;
    double fraction;
};

inline CellCoordinate locate(double value, double lower, double upper,
                      double spacing, std::size_t count) {
    if (!std::isfinite(value) || value < lower || value > upper) {
        throw std::invalid_argument("Control point is outside the surface bounds");
    }
    const double position = (value - lower) / spacing;
    const auto index = std::min(static_cast<std::size_t>(position), count - 2);
    return {index, std::clamp(position - static_cast<double>(index), 0.0, 1.0)};
}

inline GridGeometry validateSurface(const Surface& surface) {
    if (surface.nx < 2 || surface.ny < 2 ||
        surface.nx > std::numeric_limits<std::size_t>::max() / surface.ny ||
        surface.grid.size() != surface.nx * surface.ny) {
        throw std::invalid_argument("Surface needs nx, ny >= 2 and grid.size() == nx * ny");
    }
    GridGeometry grid{surface.nx, surface.ny,
                      double(surface.minx), double(surface.maxx),
                      double(surface.miny), double(surface.maxy), 0.0, 0.0};
    if (!std::isfinite(grid.minx) || !std::isfinite(grid.maxx) ||
        !std::isfinite(grid.miny) || !std::isfinite(grid.maxy) ||
        grid.maxx <= grid.minx || grid.maxy <= grid.miny) {
        throw std::invalid_argument("Surface bounds must be finite and strictly ordered");
    }
    grid.dx = (grid.maxx - grid.minx) / double(grid.nx - 1);
    grid.dy = (grid.maxy - grid.miny) / double(grid.ny - 1);
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || grid.dx <= 0 || grid.dy <= 0) {
        throw std::invalid_argument("Invalid surface spacing");
    }
    for (qreal value : surface.grid) {
        if (!std::isfinite(double(value))) {
            throw std::invalid_argument("Surface grid must contain only finite values");
        }
    }
    return grid;
}

inline double Observation::evaluate(const std::vector<double>& values) const {
    double result = 0.0;
    for (std::size_t k = 0; k < nodes.size(); ++k) {
        result += weights[k] * values[nodes[k]];
    }
    return result;
}

inline std::vector<Observation> makeObservations(const GridGeometry& grid,
                                         const std::vector<Point>& points,
                                         std::size_t maxPoints) {
    std::vector<Observation> observations;
    // Merge equal coordinates before factorization; never soften hard data.
    std::map<std::pair<double, double>, std::size_t> locations;
    for (std::size_t p = 0; p < points.size(); ++p) {
        const auto& point = points[p];
        const double target = double(point.value);
        const double weight = double(point.weight);
        if (!std::isfinite(target) || !std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("Point needs a finite value and finite nonnegative weight at index " + std::to_string(p));
        }
        const auto x = locate(double(point.x), grid.minx, grid.maxx, grid.dx, grid.nx);
        const auto y = locate(double(point.y), grid.miny, grid.maxy, grid.dy, grid.ny);
        const auto inserted = locations.emplace(std::make_pair(double(point.x), double(point.y)), observations.size());
        if (!inserted.second) {
            auto& existing = observations[inserted.first->second];
            if (existing.target != target) {
                throw std::invalid_argument("Conflicting targets at duplicate point " + std::to_string(p));
            }
            existing.relativeWeight = std::max(existing.relativeWeight, weight);
            continue;
        }
        if (observations.size() >= maxPoints) {
            throw std::runtime_error("Control point limit exceeded");
        }
        const std::size_t lowerLeft = y.index * grid.nx + x.index;
        observations.push_back({
            {lowerLeft, lowerLeft + 1, lowerLeft + grid.nx, lowerLeft + grid.nx + 1},
            {(1 - x.fraction) * (1 - y.fraction), x.fraction * (1 - y.fraction),
             (1 - x.fraction) * y.fraction, x.fraction * y.fraction},
            target, weight
        });
    }
    if (!observations.empty()) {
        double largestWeight = 0.0;
        for (const auto& point : observations) {
            largestWeight = std::max(largestWeight, point.relativeWeight);
        }
        if (largestWeight == 0.0) {
            throw std::invalid_argument("At least one control point must have positive weight");
        }
        // Scale first so that even large input weights cannot overflow the mean.
        long double sum = 0.0L;
        for (const auto& point : observations) {
            sum += point.relativeWeight / largestWeight;
        }
        const double scaledMean = double(sum / observations.size());
        for (auto& point : observations) {
            point.relativeWeight = (point.relativeWeight / largestWeight) / scaledMean;
        }
    }
    return observations;
}
// -----------------------------------------------------------------------------
// 3. Variogram model, circulant embedding and FFT generator
// -----------------------------------------------------------------------------

// PocketFFT defines this macro for its default or caller-selected namespace.
namespace fft = ::POCKETFFT_NAMESPACE;

// Unit-variance correlation field. The Simulator applies standardDeviation.
class SpectralField {
public:
    SpectralField(const GridGeometry& grid, const CovarianceParameters& parameters,
                  std::size_t maxEmbeddingNodes);

    std::vector<double> sample(std::uint64_t seed) const;
    std::vector<double> multiplyCovariance(const std::vector<double>& values) const;
    double covariance(std::size_t firstNode, std::size_t secondNode) const;

    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }

private:
    using Buffer = std::vector<std::complex<double>>;
    void transform(Buffer& buffer, bool forward) const;
    std::vector<double> crop(const Buffer& buffer) const;

    GridGeometry grid_;
    std::size_t nx_ = 0;
    std::size_t ny_ = 0;
    std::vector<double> spectrum_;
    // Effective covariance after removing only roundoff-negative eigenvalues.
    std::vector<double> kernel_;
};

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double logTwenty = 2.99573227355399099344;
// Positive root of (1 + a) * exp(-a) = 0.05.
inline constexpr double matern95 = 4.743864518390578;
inline constexpr double spectralTolerance = 1e-12;

inline void validateCovariance(const CovarianceParameters& p) {
    if (!std::isfinite(p.standardDeviation) || p.standardDeviation <= 0 ||
        !std::isfinite(p.range) || p.range <= 0 ||
        !std::isfinite(p.anisotropyRatio) || p.anisotropyRatio <= 0 || p.anisotropyRatio > 1 ||
        !std::isfinite(p.azimuthDegrees)) {
        throw std::invalid_argument("Covariance needs positive deviation and range, anisotropyRatio in (0, 1], finite angle");
    }
    const double variance = p.standardDeviation * p.standardDeviation;
    if (!std::isfinite(variance) || variance <= 0 ||
        !std::isfinite(p.nugget) || p.nugget < 0 || p.nugget > variance) {
        throw std::invalid_argument("Nugget must be in [0, standardDeviation^2], with finite positive total variance");
    }
    if (p.range * p.anisotropyRatio <= 0) {
        throw std::invalid_argument("Minor-axis range underflows");
    }
    switch (p.variogram) {
    case VariogramType::Matern32:
    case VariogramType::Exponential:
    case VariogramType::Gaussian:
    case VariogramType::Spherical:
        break;
    default:
        throw std::invalid_argument("Unknown variogram type");
    }
}

inline std::size_t expandedSize(std::size_t size) {
    if (size > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::runtime_error("FFT embedding size overflow");
    }
    return 2 * size - 1;
}

// Odd dimensions avoid ambiguous Nyquist lags for rotated anisotropy.
inline double signedLag(std::size_t index, std::size_t size, double spacing) {
    const double steps = index <= size / 2 ? double(index) : -double(size - index);
    const double lag = steps * spacing;
    if (!std::isfinite(lag)) {
        throw std::invalid_argument("FFT embedding coordinates overflow");
    }
    return lag;
}

inline double structuredCorrelation(double r, VariogramType type) {
    if (!std::isfinite(r)) {
        return 0.0;
    }
    switch (type) {
    case VariogramType::Matern32: {
        const double t = matern95 * r;
        return std::isfinite(t) ? (1.0 + t) * std::exp(-t) : 0.0;
    }
    case VariogramType::Exponential:
        return std::exp(-logTwenty * r);
    case VariogramType::Gaussian:
        return std::exp(-logTwenty * r * r);
    case VariogramType::Spherical:
        // Equivalent to 1 - 1.5*r + 0.5*r^3, stable near the cutoff.
        return r < 1.0 ? (1.0 - r) * (1.0 - r) * (1.0 + 0.5 * r) : 0.0;
    default:
        throw std::invalid_argument("Unknown variogram type");
    }
}

inline double correlationAtLag(double x, double y, const CovarianceParameters& p,
                               double cosine, double sine, double structuredFraction) {
    if (x == 0.0 && y == 0.0) {
        return 1.0; // Structured variance + independent nodal nugget.
    }
    if (structuredFraction == 0.0) {
        return 0.0; // Pure nugget: independent values at distinct grid nodes.
    }
    const double along = (cosine * x + sine * y) / p.range;
    const double across = (-sine * x + cosine * y) / (p.range * p.anisotropyRatio);
    return structuredFraction * structuredCorrelation(std::hypot(along, across), p.variogram);
}

inline SpectralField::SpectralField(const GridGeometry& grid, const CovarianceParameters& p,
                             std::size_t maxEmbeddingNodes) : grid_(grid) {
    validateCovariance(p);
    nx_ = expandedSize(grid.nx);
    ny_ = expandedSize(grid.ny);
    const double angle = std::remainder(p.azimuthDegrees, 360.0) * pi / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double variance = p.standardDeviation * p.standardDeviation;
    const double structuredFraction = (variance - p.nugget) / variance;

    for (;;) {
        if (nx_ > maxEmbeddingNodes / ny_) {
            throw std::runtime_error("No valid FFT embedding within maxEmbeddingNodes; increase limit or reduce range");
        }
        const std::size_t count = nx_ * ny_;
        if (count > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(std::complex<double>)) {
            throw std::runtime_error("FFT embedding exceeds addressable byte span");
        }
        Buffer buffer(count);
        for (std::size_t y = 0; y < ny_; ++y) {
            for (std::size_t x = 0; x < nx_; ++x) {
                buffer[y * nx_ + x] = correlationAtLag(
                    signedLag(x, nx_, grid.dx), signedLag(y, ny_, grid.dy),
                    p, cosine, sine, structuredFraction);
            }
        }
        transform(buffer, true);

        double largest = 0.0;
        double smallest = 0.0;
        double imaginary = 0.0;
        for (const auto& value : buffer) {
            largest = std::max(largest, std::abs(value.real()));
            smallest = std::min(smallest, value.real());
            imaginary = std::max(imaginary, std::abs(value.imag()));
        }
        const double tolerance = spectralTolerance * std::max(1.0, largest);
        if (imaginary > tolerance) {
            throw std::runtime_error("FFT covariance lost central symmetry");
        }
        if (smallest < -tolerance) {
            nx_ = expandedSize(nx_);
            ny_ = expandedSize(ny_);
            continue;
        }

        spectrum_.resize(count);
        // Pair opposite frequencies to preserve exact Hermitian symmetry.
        for (std::size_t y = 0; y < ny_; ++y) {
            for (std::size_t x = 0; x < nx_; ++x) {
                const std::size_t opposite = ((ny_ - y) % ny_) * nx_ + (nx_ - x) % nx_;
                spectrum_[y * nx_ + x] = std::max(0.0,
                    0.5 * (buffer[y * nx_ + x].real() + buffer[opposite].real()));
            }
        }
        long double spectrumSum = 0.0L;
        for (double value : spectrum_) {
            spectrumSum += value;
        }
        const double effectiveVariance = double(spectrumSum / count);
        if (!std::isfinite(effectiveVariance) || effectiveVariance <= 0.0) {
            throw std::runtime_error("Invalid FFT covariance variance");
        }
        for (std::size_t i = 0; i < count; ++i) {
            spectrum_[i] /= effectiveVariance;
            buffer[i] = spectrum_[i];
        }
        transform(buffer, false);
        kernel_.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            kernel_[i] = buffer[i].real();
        }
        return;
    }
}

inline void SpectralField::transform(Buffer& buffer, bool forward) const {
    const fft::shape_t shape{ny_, nx_};
    const fft::stride_t strides{
        static_cast<std::ptrdiff_t>(nx_ * sizeof(std::complex<double>)),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
    const fft::shape_t axes{0, 1};
    const double scale = forward ? 1.0 : 1.0 / double(buffer.size());
    fft::c2c(shape, strides, strides, axes, forward,
                                  buffer.data(), buffer.data(), scale, 1);
}

inline std::vector<double> SpectralField::crop(const Buffer& buffer) const {
    std::vector<double> result(grid_.nx * grid_.ny);
    for (std::size_t y = 0; y < grid_.ny; ++y) {
        for (std::size_t x = 0; x < grid_.nx; ++x) {
            result[y * grid_.nx + x] = buffer[y * nx_ + x].real();
        }
    }
    return result;
}

inline std::vector<double> SpectralField::sample(std::uint64_t seed) const {
    std::mt19937_64 random(seed);
    std::normal_distribution<double> normal;
    Buffer buffer(spectrum_.size());
    for (auto& value : buffer) {
        value = normal(random);
    }
    // Real white noise -> FFT -> sqrt(spectrum) -> inverse FFT / M.
    transform(buffer, true);
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] *= std::sqrt(spectrum_[i]);
    }
    transform(buffer, false);
    return crop(buffer);
}

inline std::vector<double> SpectralField::multiplyCovariance(const std::vector<double>& values) const {
    Buffer buffer(spectrum_.size(), 0.0);
    for (std::size_t y = 0; y < grid_.ny; ++y) {
        for (std::size_t x = 0; x < grid_.nx; ++x) {
            buffer[y * nx_ + x] = values[y * grid_.nx + x];
        }
    }
    transform(buffer, true);
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] *= spectrum_[i];
    }
    transform(buffer, false);
    return crop(buffer);
}

inline double SpectralField::covariance(std::size_t a, std::size_t b) const {
    const std::size_t ax = a % grid_.nx, ay = a / grid_.nx;
    const std::size_t bx = b % grid_.nx, by = b / grid_.nx;
    const std::size_t x = ax >= bx ? ax - bx : nx_ - (bx - ax);
    const std::size_t y = ay >= by ? ay - by : ny_ - (by - ay);
    return kernel_[y * nx_ + x];
}

} // namespace gaussian_surface::detail

// -----------------------------------------------------------------------------
// 4. Gaussian conditioning and public method definitions
// -----------------------------------------------------------------------------

namespace gaussian_surface {

struct Simulator::Impl {
    Surface base;
    std::vector<double> mean;
    double deviation;
    std::vector<detail::Observation> observations;
    detail::SpectralField field;
    Eigen::LLT<Eigen::MatrixXd> factor;
    Eigen::VectorXd standardizedTargets;

    Impl(const Surface& surface, const std::vector<Point>& points,
         const CovarianceParameters& covariance, const Limits& limits)
        : base(surface), mean(surface.grid.begin(), surface.grid.end()),
          deviation(covariance.standardDeviation),
          observations(detail::makeObservations(detail::validateSurface(surface),
                                                 points, limits.maxControlPoints)),
          field(detail::validateSurface(surface), covariance, limits.maxEmbeddingNodes) {
        prepareConditioning();
    }

    Eigen::Vector3d trendBasis(std::size_t node) const {
        return {1.0,
                2.0 * double(node % base.nx) / double(base.nx - 1) - 1.0,
                2.0 * double(node / base.nx) / double(base.ny - 1) - 1.0};
    }

    void applyWeightedTrend() {
        std::size_t activeCount = 0;
        double firstWeight = 0.0;
        bool equalWeights = true;
        for (const auto& point : observations) {
            if (point.relativeWeight == 0.0) continue;
            if (activeCount == 0) firstWeight = point.relativeWeight;
            else if (point.relativeWeight != firstWeight) equalWeights = false;
            ++activeCount;
        }
        if (equalWeights) return; // Includes a single active point; preserve the base exactly.

        const auto count = static_cast<Eigen::Index>(activeCount);
        Eigen::MatrixXd design(count, 3);
        Eigen::VectorXd residuals(count), influence(count);
        Eigen::Index row = 0;
        for (const auto& point : observations) {
            if (point.relativeWeight == 0.0) continue;
            design.row(row).setZero();
            for (std::size_t k = 0; k < 4; ++k) {
                design.row(row) += point.weights[k] * trendBasis(point.nodes[k]).transpose();
            }
            residuals[row] = point.target - point.evaluate(mean);
            influence[row] = point.relativeWeight;
            ++row;
        }
        if (!residuals.allFinite()) {
            throw std::invalid_argument("Control residual overflows during trend preparation");
        }
        const double residualScale = residuals.cwiseAbs().maxCoeff();
        if (residualScale == 0.0) return;
        residuals /= residualScale;

        // WLS minimizes sum(q_i * error_i^2), hence sqrt(q_i) scales the rows.
        Eigen::MatrixXd weightedDesign = design;
        Eigen::VectorXd weightedResiduals = residuals;
        for (Eigen::Index i = 0; i < count; ++i) {
            const double scale = std::sqrt(influence[i]);
            weightedDesign.row(i) *= scale;
            weightedResiduals[i] *= scale;
        }
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> weightedFit(weightedDesign);
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> equalFit(design);
        weightedFit.setThreshold(1e-10);
        equalFit.setThreshold(1e-10);
        Eigen::Vector3d delta = Eigen::Vector3d::Zero();
        if (weightedFit.rank() == 3 && equalFit.rank() == 3) {
            delta = weightedFit.solve(weightedResiduals) - equalFit.solve(residuals);
        } else {
            // Both fits use the same constant basis when a plane cannot be resolved.
            delta[0] = influence.dot(residuals) / influence.sum() - residuals.mean();
        }
        delta *= residualScale;
        if (!delta.allFinite()) {
            throw std::runtime_error("Weighted trend coefficients overflow");
        }
        for (std::size_t i = 0; i < mean.size(); ++i) {
            mean[i] += trendBasis(i).dot(delta);
            if (!std::isfinite(mean[i])) {
                throw std::runtime_error("Weighted mean surface overflows");
            }
        }
    }

    void prepareConditioning() {
        if (observations.size() > std::size_t(std::numeric_limits<Eigen::Index>::max())) {
            throw std::runtime_error("Too many observations for Eigen");
        }
        const auto count = static_cast<Eigen::Index>(observations.size());
        standardizedTargets.resize(count);
        if (count == 0) {
            return;
        }
        applyWeightedTrend();
        Eigen::MatrixXd matrix(count, count);
        for (Eigen::Index i = 0; i < count; ++i) {
            const auto& a = observations[std::size_t(i)];
            standardizedTargets[i] = (a.target - a.evaluate(mean)) / deviation;
            for (Eigen::Index j = 0; j <= i; ++j) {
                const auto& b = observations[std::size_t(j)];
                double value = 0.0;
                // H * C * H^T: the same grid covariance as the FFT sampler.
                for (std::size_t u = 0; u < 4; ++u) {
                    for (std::size_t v = 0; v < 4; ++v) {
                        value += a.weights[u] * b.weights[v] * field.covariance(a.nodes[u], b.nodes[v]);
                    }
                }
                matrix(i, j) = matrix(j, i) = value;
            }
        }
        if (!standardizedTargets.allFinite()) {
            throw std::invalid_argument("Control residual divided by deviation overflows");
        }
        factor.compute(matrix);
        if (factor.info() != Eigen::Success || !std::isfinite(factor.rcond()) || factor.rcond() < 1e-12) {
            throw std::runtime_error("Dependent or ill-conditioned control constraints; remove redundant points or adjust covariance");
        }
    }

    Surface generate(std::uint64_t seed) const {
        auto residual = field.sample(seed);
        if (!observations.empty()) {
            Eigen::VectorXd difference = standardizedTargets;
            for (std::size_t i = 0; i < observations.size(); ++i) {
                difference[static_cast<Eigen::Index>(i)] -= observations[i].evaluate(residual);
            }
            const Eigen::VectorXd coefficients = factor.solve(difference);
            if (!coefficients.allFinite()) {
                throw std::runtime_error("Non-finite conditioning solution");
            }
            std::vector<double> scattered(mean.size(), 0.0);
            for (std::size_t i = 0; i < observations.size(); ++i) {
                const auto& point = observations[i];
                for (std::size_t k = 0; k < 4; ++k) {
                    scattered[point.nodes[k]] += point.weights[k] * coefficients[static_cast<Eigen::Index>(i)];
                }
            }
            const auto correction = field.multiplyCovariance(scattered);
            for (std::size_t i = 0; i < residual.size(); ++i) {
                residual[i] += correction[i];
            }
        }

        Surface result = base;
        for (std::size_t i = 0; i < residual.size(); ++i) {
            const double value = mean[i] + deviation * residual[i];
            if (!std::isfinite(value) || std::abs(value) > double(std::numeric_limits<qreal>::max())) {
                throw std::runtime_error("Generated value is not representable as qreal");
            }
            result.grid[i] = static_cast<qreal>(value);
            residual[i] = double(result.grid[i]); // Validate the returned representation.
        }
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const auto& point = observations[i];
            double scale = std::max(1.0, std::abs(point.target));
            for (std::size_t k = 0; k < 4; ++k) {
                scale = std::max(scale, std::abs(residual[point.nodes[k]]));
            }
            const double tolerance = 1e-8 + std::max(1e-10, 32.0 * double(std::numeric_limits<qreal>::epsilon())) * scale;
            if (std::abs(point.evaluate(residual) - point.target) > tolerance) {
                throw std::runtime_error("Hard conditioning tolerance exceeded at point " + std::to_string(i));
            }
        }
        return result;
    }
};

inline Simulator::Simulator(const Surface& surface, const std::vector<Point>& points,
                     const CovarianceParameters& covariance, const Limits& limits)
    : impl_(std::make_shared<Impl>(surface, points, covariance, limits)) {}

inline Surface Simulator::generate(std::uint64_t seed) const {
    if (!impl_) {
        throw std::logic_error("Cannot use a moved-from Simulator");
    }
    return impl_->generate(seed);
}

inline PreparationInfo Simulator::info() const {
    if (!impl_) {
        throw std::logic_error("Cannot use a moved-from Simulator");
    }
    return {impl_->field.nx(), impl_->field.ny(), impl_->observations.size()};
}

} // namespace gaussian_surface

