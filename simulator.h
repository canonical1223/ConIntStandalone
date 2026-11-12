#pragma once

// Header-only C++17. Keep dependency configuration consistent across all .cpp files.
// Dependency configuration belongs to the consuming application.
#include <QtGlobal>
#include <Eigen/Cholesky>
#include <Eigen/Core>
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
    // Exact target = value * weight. Zero weight means target zero.
    qreal weight = 1;
};

struct CovarianceParameters {
    double standardDeviation = 1.0;
    // Matern 3/2 correlation: (1 + sqrt(3)*r)*exp(-sqrt(3)*r).
    // These are length scales, NOT distances where correlation becomes zero.
    double majorLength = 1.0;
    double minorLength = 1.0;
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
    std::map<std::pair<double, double>, double> targets;
    for (std::size_t p = 0; p < points.size(); ++p) {
        const auto& point = points[p];
        const double target = double(point.value) * double(point.weight);
        if (!std::isfinite(double(point.value)) ||
            !std::isfinite(double(point.weight)) || !std::isfinite(target)) {
            throw std::invalid_argument("Non-finite value * weight at point " + std::to_string(p));
        }
        const auto x = locate(double(point.x), grid.minx, grid.maxx, grid.dx, grid.nx);
        const auto y = locate(double(point.y), grid.miny, grid.maxy, grid.dy, grid.ny);
        const auto inserted = targets.emplace(std::make_pair(double(point.x), double(point.y)), target);
        if (!inserted.second) {
            if (inserted.first->second != target) {
                throw std::invalid_argument("Conflicting targets at duplicate point " + std::to_string(p));
            }
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
            target
        });
    }
    return observations;
}
// -----------------------------------------------------------------------------
// 3. Covariance model, circulant embedding and FFT generator
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
inline constexpr double sqrtThree = 1.73205080756887729353;
inline constexpr double spectralTolerance = 1e-12;

inline void validateCovariance(const CovarianceParameters& p) {
    if (!std::isfinite(p.standardDeviation) || p.standardDeviation <= 0 ||
        !std::isfinite(p.majorLength) || p.majorLength <= 0 ||
        !std::isfinite(p.minorLength) || p.minorLength <= 0 ||
        !std::isfinite(p.azimuthDegrees)) {
        throw std::invalid_argument("Covariance needs finite positive deviation and lengths, finite angle");
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

inline double maternCorrelation(double x, double y, const CovarianceParameters& p,
                         double cosine, double sine) {
    const double along = (cosine * x + sine * y) / p.majorLength;
    const double across = (-sine * x + cosine * y) / p.minorLength;
    const double r = sqrtThree * std::hypot(along, across);
    return std::isfinite(r) ? (1.0 + r) * std::exp(-r) : 0.0;
}

inline SpectralField::SpectralField(const GridGeometry& grid, const CovarianceParameters& p,
                             std::size_t maxEmbeddingNodes) : grid_(grid) {
    validateCovariance(p);
    nx_ = expandedSize(grid.nx);
    ny_ = expandedSize(grid.ny);
    const double angle = std::remainder(p.azimuthDegrees, 360.0) * pi / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);

    for (;;) {
        if (nx_ > maxEmbeddingNodes / ny_) {
            throw std::runtime_error("No valid FFT embedding within maxEmbeddingNodes; increase limit or shorten lengths");
        }
        const std::size_t count = nx_ * ny_;
        if (count > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(std::complex<double>)) {
            throw std::runtime_error("FFT embedding exceeds addressable byte span");
        }
        Buffer buffer(count);
        for (std::size_t y = 0; y < ny_; ++y) {
            for (std::size_t x = 0; x < nx_; ++x) {
                buffer[y * nx_ + x] = maternCorrelation(
                    signedLag(x, nx_, grid.dx), signedLag(y, ny_, grid.dy), p, cosine, sine);
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
        const double variance = double(spectrumSum / count);
        if (!std::isfinite(variance) || variance <= 0.0) {
            throw std::runtime_error("Invalid FFT covariance variance");
        }
        for (std::size_t i = 0; i < count; ++i) {
            spectrum_[i] /= variance;
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

    void prepareConditioning() {
        if (observations.size() > std::size_t(std::numeric_limits<Eigen::Index>::max())) {
            throw std::runtime_error("Too many observations for Eigen");
        }
        const auto count = static_cast<Eigen::Index>(observations.size());
        standardizedTargets.resize(count);
        if (count == 0) {
            return;
        }
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

