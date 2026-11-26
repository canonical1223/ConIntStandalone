#ifndef GAUSSIAN_SURFACE_SURFACE_GRFS_HPP
#define GAUSSIAN_SURFACE_SURFACE_GRFS_HPP

// Independent C++17 conditional Gaussian simulator for a regular XY node grid.
// One Simulator prepares an embedding and a global kriging factor; generate(seed)
// is const and may be called concurrently. Elevation is bilinearly interpolated
// at the original observation XY. No other project headers are required.
//
//   using namespace surface_grfs;
//   Grid grid{201, 151, 0, 0, 25, 25};
//   Covariance cov{0.5, {{Kernel::spherical, 8.5, 400, 180, 30}}};
//   Options opt; opt.mean = -1200.0;
//   Simulator sim(grid, cov, {{173, 418, -1195}, {825, 310, -1203}}, {}, opt);
//   auto z = sim.generate(42);           // z[j * grid.nx + i], elevation units
//
// Method: u ~ N(0, C), z = m + u + C H' (H C H')^-1 (d - H(m+u)).
// H interpolates from grid nodes. The nugget belongs to nodal variance. Exact
// observations, one stationary domain, no measurement noise. Ill-conditioned
// constraints or an infeasible embedding are reported by exception.
// Petrel comparison: Daly, Quental & Novak, A Faster, More Accurate Gaussian
// Simulation (GeoCanada 2010). This API implements independent numerical choices.
// Theory: Dietrich & Newsam, doi:10.1029/94WR02977. See docs/surface-grfs.md.
//
// Heights are absolute. Pass a row-major mean/trend grid as argument four, or
// set Options::mean. Without either, use the mean of the unique observations
// (zero with no observations). The nugget and partial sills are variances in
// squared height units. Noncompact ranges give 5% remaining correlation.
// XY samples with identical elevations at the same XY are merged. Constraints
// must be independent as linear combinations of grid nodes.
//
// For M embedding nodes and p observations: preparation per attempted FFT
// embedding is O(M log M); one dense Cholesky factorization is O(p^3).
// The general draw costs O(M log M + p^2). Compact-support corrections can
// replace two FFTs with a spatial pass. info() reports the chosen geometry.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace surface_grfs {

struct Grid {
    std::size_t nx, ny;
    double x0, y0, dx, dy;  // first node, positive spacing, X east, Y north
};

struct Point { double x, y, value; };
enum class Kernel { spherical, exponential, gaussian, matern32 };

struct Structure {
    Kernel kernel;
    double partial_sill;             // squared elevation units
    double major_range, minor_range; // same units as XY
    double azimuth_deg = 0;          // clockwise from north
};

struct Covariance {
    double nugget = 0;
    std::vector<Structure> structures{{Kernel::spherical, 1, 1, 1, 0}};
};

struct Options {
    // Used without a trend grid. Empty = mean of distinct hard data (0 if none).
    std::optional<double> mean;
    std::size_t fft_threads = 1;
    std::size_t max_embedding_nodes = 16 * 1024 * 1024;
    std::size_t max_control_points = 2048;
};

struct PreparationInfo {
    std::size_t embedding_nx, embedding_ny, control_points;
    bool compact_correction; // spatial compact-support convolution instead of extra FFTs
};

namespace detail {

inline void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}

inline Grid validate_grid(Grid grid, const Options& options) {
    require(options.fft_threads > 0 && options.max_embedding_nodes > 0,
            "GRFS FFT thread count and embedding limit must be positive");
    require(!options.mean || std::isfinite(*options.mean), "GRFS invalid mean");
    auto check_axis = [](std::size_t n, double first, double step) {
        require(n > 0 && n < static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()) / 4 &&
                static_cast<double>(n - 1) < 0x1p52, "GRFS invalid grid dimensions");
        require(std::isfinite(first) && std::isfinite(step) && step > 0, "GRFS invalid grid geometry");
        const double extent = static_cast<double>(n - 1) * step;
        const double last = std::fma(static_cast<double>(n - 1), step, first);
        require(std::isfinite(extent) && std::isfinite(last), "GRFS grid extent overflow");
        if (n > 1) {
            const double magnitude = (std::max)(std::abs(first), std::abs(last));
            require(last > first && step >= magnitude - std::nextafter(magnitude, 0.),
                    "GRFS adjacent nodes are unresolved in world coordinates");
        }
    };
    check_axis(grid.nx, grid.x0, grid.dx);
    check_axis(grid.ny, grid.y0, grid.dy);
    if (grid.nx > options.max_embedding_nodes / grid.ny)
        throw std::length_error("GRFS grid exceeds embedding limit");
    return grid;
}

// Unit-variance covariance. For asymptotic kernels the structural correlation
// at the practical range is 0.05. Spherical structures have finite support.
class Model {
    struct Term {
        Structure spec;
        double sine, cosine;
    };
    double dx_, dy_, sill_;
    std::vector<Term> terms_;
    std::size_t radius_x_ = 0, radius_y_ = 0;
    bool compact_ = true;

    static std::size_t support_steps(double extent, double spacing) {
        const double steps = std::ceil(extent / spacing);
        constexpr auto cap = (std::numeric_limits<std::ptrdiff_t>::max)() / 4;
        if (!(steps < static_cast<double>(cap))) return static_cast<std::size_t>(cap);
        return static_cast<std::size_t>(steps);
    }
public:
    Model(const Grid& grid, const Covariance& covariance)
        : dx_(grid.dx), dy_(grid.dy), sill_(covariance.nugget) {
        require(std::isfinite(sill_) && sill_ >= 0, "GRFS invalid nugget");
        for (const auto& s : covariance.structures) {
            require(std::isfinite(s.partial_sill) && s.partial_sill >= 0 &&
                    std::isfinite(s.major_range) && s.major_range > 0 &&
                    std::isfinite(s.minor_range) && s.minor_range > 0 &&
                    std::isfinite(s.azimuth_deg), "GRFS invalid covariance structure");
            require(s.kernel == Kernel::spherical || s.kernel == Kernel::exponential ||
                    s.kernel == Kernel::gaussian || s.kernel == Kernel::matern32, "GRFS unknown kernel");
            if (s.partial_sill == 0) continue;
            const double angle = std::remainder(s.azimuth_deg, 360.) * 0.01745329251994329577;
            const double sine = std::sin(angle), cosine = std::cos(angle);
            terms_.push_back({s, sine, cosine});
            sill_ += s.partial_sill;
            // Exact support bounds for a spherical term, practical ranges for
            // other terms. These also guide extension of an anisotropic embedding.
            radius_x_ = (std::max)(radius_x_, support_steps(
                std::hypot(s.major_range * sine, s.minor_range * cosine), dx_));
            radius_y_ = (std::max)(radius_y_, support_steps(
                std::hypot(s.major_range * cosine, s.minor_range * sine), dy_));
            if (s.kernel != Kernel::spherical) compact_ = false;
        }
        require(std::isfinite(sill_) && sill_ > 0, "GRFS total sill must be finite and positive");
        for (auto& t : terms_) t.spec.partial_sill /= sill_;
    }
    double sill() const { return sill_; }
    bool compact() const { return compact_; }
    std::size_t radius_x() const { return radius_x_; }
    std::size_t radius_y() const { return radius_y_; }
    double correlation(std::ptrdiff_t hx, std::ptrdiff_t hy) const {
        if (hx == 0 && hy == 0) return 1.;
        const double x = static_cast<double>(hx) * dx_, y = static_cast<double>(hy) * dy_;
        if (!std::isfinite(x) || !std::isfinite(y))
            throw std::overflow_error("GRFS embedding lag overflow");
        double result = 0.;
        for (const auto& t : terms_) {
            const double a = (t.sine * x + t.cosine * y) / t.spec.major_range;
            const double b = (t.cosine * x - t.sine * y) / t.spec.minor_range;
            const double r2 = a*a + b*b;
            if (std::isnan(r2)) throw std::overflow_error("GRFS rotated lag overflow");
            constexpr double log20 = 2.9957322735539909934;
            double rho;
            switch (t.spec.kernel) {
                case Kernel::spherical: {
                    if (r2 >= 1.) continue;
                    const double r = std::sqrt(r2);
                    rho = (1.-r) * (1.-r) * (1.+0.5*r);
                    break;
                }
                case Kernel::exponential: rho = std::exp(-log20 * std::sqrt(r2)); break;
                case Kernel::gaussian: rho = std::exp(-log20 * r2); break;
                case Kernel::matern32: {
                    const double v = 4.743864518390578 * std::sqrt(r2);
                    rho = std::isfinite(v) ? (1.+v) * std::exp(-v) : 0.;
                    break;
                }
                default: throw std::logic_error("GRFS unknown kernel");
            }
            result += t.spec.partial_sill * rho;
        }
        return result;
    }
};

struct SupportNode { std::size_t x, y, index; double weight; };
struct Constraint {
    double target;
    std::array<SupportNode, 4> nodes{};
    std::size_t count = 0;
    double evaluate(const std::vector<double>& field) const {
        double z = 0;
        for (std::size_t i = 0; i < count; ++i) z += nodes[i].weight * field[nodes[i].index];
        return z;
    }
};

struct AxisLocation { std::size_t lo, hi; double fraction; };
inline AxisLocation locate(double value, double first, double step, std::size_t n) {
    const double last = std::fma(static_cast<double>(n - 1), step, first);
    require(std::isfinite(value) && value >= first && value <= last, "GRFS point outside grid");
    if (value == last) return {n-1, n-1, 0};
    const double pos = (value - first) / step;
    // A representable world coordinate of a node must recover that node even
    // when a large map origin loses low-order bits of the nominal offset.
    const auto nearest = (std::min)(static_cast<std::size_t>(std::round(pos)), n-1);
    if (value == std::fma(static_cast<double>(nearest), step, first))
        return {nearest, nearest, 0};
    const auto lo = (std::min)(static_cast<std::size_t>(pos), n-2);
    return {lo, lo+1, std::clamp(pos-static_cast<double>(lo), 0., 1.)};
}

inline std::vector<Constraint> prepare_points(const Grid& g, const std::vector<Point>& points,
                                              std::size_t limit) {
    std::map<std::pair<double,double>, double> unique;
    for (const auto& p : points) {
        require(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.value), "GRFS nonfinite observation");
        const auto item = unique.emplace(std::make_pair(p.x,p.y), p.value);
        require(item.second || item.first->second == p.value, "GRFS conflicting colocated observations");
        if (unique.size() > limit) throw std::length_error("GRFS control point limit exceeded");
    }
    std::vector<Constraint> result;
    result.reserve(unique.size());
    for (const auto& p : unique) {
        const auto x = locate(p.first.first, g.x0, g.dx, g.nx);
        const auto y = locate(p.first.second, g.y0, g.dy, g.ny);
        Constraint c{p.second};
        auto add = [&](std::size_t i, std::size_t j, double weight) {
            if (weight > 0) c.nodes[c.count++] = {i,j,i + j*g.nx,weight};
        };
        add(x.lo, y.lo, (1.-x.fraction)*(1.-y.fraction));
        add(x.hi, y.lo, x.fraction*(1.-y.fraction));
        add(x.lo, y.hi, (1.-x.fraction)*y.fraction);
        add(x.hi, y.hi, x.fraction*y.fraction);
        result.push_back(c);
    }
    return result;
}

// Iterative radix-2 FFT. Both directions are unscaled; callers choose the
// normalization for covariance convolution and spectral Gaussian draws.
class SurfaceFftPlan {
    using Complex = std::complex<double>;
    std::vector<std::size_t> reverse_;
    std::vector<Complex> forward_roots_;

    template<bool Inverse>
    void apply(Complex* row) const {
        const std::size_t n = reverse_.size();
        for (std::size_t j = 1; j < n; ++j)
            if (j < reverse_[j]) std::swap(row[j], row[reverse_[j]]);
        // The length-two butterflies are real additions in both directions.
        for (std::size_t j = 0; j + 1 < n; j += 2) {
            const Complex a = row[j], b = row[j + 1];
            row[j] = a + b;
            row[j + 1] = a - b;
        }
        for (std::size_t half = 2; half < n; half *= 2) {
            const std::size_t width = 2 * half;
            const std::size_t stride = n / width;
            for (std::size_t block = 0; block < n; block += width) {
                for (std::size_t k = 0; k < half; ++k) {
                    const Complex w = Inverse ? std::conj(forward_roots_[k * stride])
                                              : forward_roots_[k * stride];
                    const Complex u = row[block + k];
                    const Complex v = row[block + half + k] * w;
                    row[block + k] = u + v;
                    row[block + half + k] = u - v;
                }
            }
        }
    }

public:
    void reset(std::size_t n) {
        reverse_.resize(n);
        reverse_[0] = 0;
        for (std::size_t i = 1; i < n; ++i)
            reverse_[i] = (reverse_[i / 2] >> 1) + ((i & 1) ? n / 2 : 0);
        forward_roots_.resize(n / 2);
        constexpr double tau = 6.283185307179586476925286766559;
        for (std::size_t i = 0; i < n / 2; ++i) {
            const double phase = -tau * static_cast<double>(i) / static_cast<double>(n);
            forward_roots_[i] = {std::cos(phase), std::sin(phase)};
        }
    }
    void transform(Complex* row, bool inverse) const {
        if (inverse) apply<true>(row);
        else apply<false>(row);
    }
};

class Spectral {
    using Complex = std::complex<double>;
    using Buffer = std::vector<Complex>;
    std::size_t grid_x_, grid_y_, mx_, my_, workers_;
    SurfaceFftPlan fft_x_, fft_y_;
    std::vector<double> amplitudes_; // square roots of the normalized eigenvalues
    std::vector<double> kernel_;     // lag table of the effective covariance
    bool clipped_ = false;

    static std::size_t initial_length(std::size_t n, bool compact, std::size_t radius) {
        if (n == 0) throw std::invalid_argument("GRFS empty grid dimension");
        // Full embedding: 2n-1. With finite support, the nearest periodic image
        // is beyond the support for every lag within the cropped grid if m>=n+r.
        const std::size_t pad = compact ? (std::min)(n - 1, radius) : n - 1;
        const auto limit = static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
        if (n > limit || pad > limit - n) throw std::length_error("GRFS embedding dimension overflow");
        const auto required = n + pad;
        std::size_t m = 1;
        while (m < required) {
            if (m > limit / 2) throw std::length_error("GRFS radix-2 dimension overflow");
            m *= 2;
        }
        return m;
    }
    static std::ptrdiff_t signed_lag(std::size_t pos, std::size_t n) {
        if (pos <= n / 2) return static_cast<std::ptrdiff_t>(pos);
        return -static_cast<std::ptrdiff_t>(n - pos);
    }
    static std::size_t wrap(std::ptrdiff_t lag, std::size_t n) {
        // Unsigned conversion followed by a mask is a modulo operation for
        // these power-of-two dimensions, including negative signed lags.
        return static_cast<std::size_t>(lag) & (n - 1);
    }

    template<class Function>
    void batches(std::size_t count, const Function& fn) const {
        const std::size_t nt = (std::min)(workers_, count);
        if (nt <= 1 || mx_ * my_ < 32768) {
            fn(0, count);
            return;
        }
        std::vector<std::thread> threads;
        threads.reserve(nt - 1);
        // FFT and transpose always operate on disjoint rows and have no
        // throwing operations. Join even if startup of a later worker fails.
        const std::size_t base = count / nt, extra = count % nt;
        const auto start = [&](std::size_t t) { return t * base + (std::min)(t, extra); };
        try {
            for (std::size_t t = 1; t < nt; ++t)
                threads.emplace_back([&, t] { fn(start(t), start(t + 1)); });
            fn(0, start(1));
        } catch (...) {
            for (auto& thread : threads) thread.join();
            throw;
        }
        for (auto& thread : threads) thread.join();
    }

    void rows(Buffer& data, std::size_t width, std::size_t height,
              const SurfaceFftPlan& plan, bool inverse) const {
        batches(height, [&](std::size_t begin, std::size_t end) {
            for (std::size_t row = begin; row < end; ++row)
                plan.transform(data.data() + row * width, inverse);
        });
    }
    void transpose(const Buffer& src, Buffer& dst, std::size_t width, std::size_t height) const {
        constexpr std::size_t tile = 32;
        const std::size_t strips = 1 + (width - 1) / tile;
        batches(strips, [&](std::size_t begin, std::size_t end) {
            for (std::size_t strip = begin; strip < end; ++strip) {
                const auto x0 = strip * tile;
                const auto x1 = (std::min)(x0 + tile, width);
                for (std::size_t y0 = 0; y0 < height; y0 += tile) {
                    const auto y1 = (std::min)(y0 + tile, height);
                    for (std::size_t y = y0; y < y1; ++y)
                        for (std::size_t x = x0; x < x1; ++x)
                            dst[x * height + y] = src[y * width + x];
                }
            }
        });
    }
    void transform(Buffer& data, bool inverse) const {
        if (my_ == 1) { fft_x_.transform(data.data(), inverse); return; }
        if (mx_ == 1) { fft_y_.transform(data.data(), inverse); return; }
        rows(data, mx_, my_, fft_x_, inverse);
        Buffer scratch(data.size());
        transpose(data, scratch, mx_, my_);
        rows(scratch, my_, mx_, fft_y_, inverse);
        transpose(scratch, data, my_, mx_);
    }
    std::size_t opposite(std::size_t x, std::size_t y) const {
        return ((my_ - y) & (my_ - 1)) * mx_ + ((mx_ - x) & (mx_ - 1));
    }
    bool build(const Model& model) {
        const std::size_t count = mx_ * my_;
        fft_x_.reset(mx_);
        fft_y_.reset(my_);
        Buffer freq(count);
        kernel_.resize(count);
        for (std::size_t y = 0; y < my_; ++y) {
            const auto hy = signed_lag(y, my_);
            for (std::size_t x = 0; x < mx_; ++x) {
                const auto hx = signed_lag(x, mx_);
                double c = model.correlation(hx, hy);
                // A Nyquist lag is shared by two opposite directions. Their
                // average restores central symmetry for rotated anisotropy.
                if ((mx_ > 1 && x == mx_ / 2) || (my_ > 1 && y == my_ / 2))
                    c = 0.5 * (c + model.correlation(-hx, hy));
                freq[y * mx_ + x] = kernel_[y * mx_ + x] = c;
            }
        }
        transform(freq, false);
        double peak = 0, smallest = 0, imaginary = 0;
        for (const auto& z : freq) {
            if (!(std::isfinite(z.real()) && std::isfinite(z.imag())))
                throw std::runtime_error("GRFS nonfinite covariance spectrum");
            peak = (std::max)(peak, std::abs(z.real()));
            smallest = (std::min)(smallest, z.real());
            imaginary = (std::max)(imaginary, std::abs(z.imag()));
        }
        const double tolerance = 1e-12 * (std::max)(1.0, peak);
        if (imaginary > tolerance) throw std::runtime_error("GRFS covariance lost central symmetry");
        if (smallest < -tolerance) return false;
        amplitudes_.resize(count);
        long double sum = 0;
        bool clipped = false;
        for (std::size_t y = 0; y < my_; ++y) {
            for (std::size_t x = 0; x < mx_; ++x) {
                const auto i = y * mx_ + x, j = opposite(x, y);
                const double lambda = 0.5 * (freq[i].real() + freq[j].real());
                clipped = clipped || lambda < 0;
                amplitudes_[i] = (std::max)(0.0, lambda);
                sum += amplitudes_[i];
            }
        }
        const double variance = static_cast<double>(sum / static_cast<long double>(count));
        if (!(std::isfinite(variance) && variance > 0)) throw std::runtime_error("GRFS invalid spectral variance");
        for (std::size_t i = 0; i < count; ++i) {
            const double lambda = amplitudes_[i] / variance;
            amplitudes_[i] = std::sqrt(lambda);
            if (clipped) freq[i] = lambda;
        }
        clipped_ = clipped;
        if (clipped) {
            // Use precisely the covariance of the numerical sampler when
            // roundoff-negative eigenvalues have been removed.
            transform(freq, true);
            const double scale = 1.0 / static_cast<double>(count);
            for (std::size_t i = 0; i < count; ++i) kernel_[i] = freq[i].real() * scale;
        } else {
            for (double& c : kernel_) c /= variance;
        }
        return true;
    }

    std::vector<double> crop(const Buffer& buffer, double scale) const {
        std::vector<double> result(grid_x_ * grid_y_);
        for (std::size_t y = 0; y < grid_y_; ++y)
            for (std::size_t x = 0; x < grid_x_; ++x)
                result[y * grid_x_ + x] = buffer[y * mx_ + x].real() * scale;
        return result;
    }

public:
    Spectral(const Grid& grid, const Model& model, const Options& opt)
        : grid_x_(grid.nx), grid_y_(grid.ny),
          mx_(initial_length(grid.nx, model.compact(), model.radius_x())),
          my_(initial_length(grid.ny, model.compact(), model.radius_y())),
          workers_(opt.fft_threads) {
        if (!workers_) throw std::invalid_argument("GRFS fft_threads must be positive");
        const auto hw = std::thread::hardware_concurrency();
        if (hw) workers_ = (std::min)(workers_, static_cast<std::size_t>(hw));
        const std::size_t usable = (std::min)(opt.max_embedding_nodes, Buffer{}.max_size());
        for (;;) {
            if (mx_ > usable / my_)
                throw std::length_error("GRFS embedding exceeds max_embedding_nodes; inspect range and geometry");
            if (build(model)) break;
            if ((grid_x_ == 1 && grid_y_ == 1) || mx_ > usable / my_ / 2)
                throw std::runtime_error("GRFS spectral embedding search reached max_embedding_nodes");
            // Extend the most strongly correlated axis first. Long, narrow
            // structures can need many more lags in one direction than the other.
            const double xr = static_cast<double>(model.radius_x()) / static_cast<double>(mx_);
            const double yr = static_cast<double>(model.radius_y()) / static_cast<double>(my_);
            if (grid_y_ == 1 || (grid_x_ > 1 && xr >= yr)) mx_ *= 2;
            else my_ *= 2;
        }
    }
    std::size_t nx() const { return mx_; }
    std::size_t ny() const { return my_; }
    bool clipped() const { return clipped_; }
    double correlation(std::ptrdiff_t hx, std::ptrdiff_t hy) const {
        return kernel_[wrap(hy, my_) * mx_ + wrap(hx, mx_)];
    }
    std::vector<double> sample(std::uint64_t seed) const {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> normal;
        Buffer freq(mx_ * my_);
        constexpr double inv_sqrt2 = 0.70710678118654752440084436210485;
        for (std::size_t y = 0; y < my_; ++y) {
            for (std::size_t x = 0; x < mx_; ++x) {
                const auto i = y * mx_ + x, j = opposite(x, y);
                if (i > j) continue;
                const double a = normal(rng) * amplitudes_[i];
                if (i == j) {
                    freq[i] = a;
                } else {
                    const double b = normal(rng) * amplitudes_[i];
                    freq[i] = Complex(a * inv_sqrt2, b * inv_sqrt2);
                    freq[j] = std::conj(freq[i]);
                }
            }
        }
        transform(freq, true);
        return crop(freq, 1.0 / std::sqrt(static_cast<double>(freq.size())));
    }
    std::vector<double> convolve(const std::vector<double>& cropped) const {
        if (cropped.size() != grid_x_ * grid_y_) throw std::invalid_argument("GRFS incorrect convolution input size");
        Buffer freq(mx_ * my_);
        for (std::size_t y = 0; y < grid_y_; ++y)
            for (std::size_t x = 0; x < grid_x_; ++x) freq[y * mx_ + x] = cropped[y * grid_x_ + x];
        transform(freq, false);
        for (std::size_t i = 0; i < freq.size(); ++i) freq[i] *= amplitudes_[i] * amplitudes_[i];
        transform(freq, true);
        return crop(freq, 1.0 / static_cast<double>(freq.size()));
    }
};

// Cholesky factor of H C H^T, using the same unit covariance as the sampler.
// The upper triangle is unused. H has at most four nonzero entries per row.
inline std::vector<double> factor_constraints(const std::vector<Constraint>& points,
                                              const Spectral& spectral) {
    const auto n = points.size();
    if (n && n > std::vector<double>{}.max_size() / n)
        throw std::length_error("GRFS control matrix size overflow");
    std::vector<double> lower(n*n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double entry = 0;
            for (std::size_t a = 0; a < points[i].count; ++a) {
                const auto& u = points[i].nodes[a];
                for (std::size_t b = 0; b < points[j].count; ++b) {
                    const auto& v = points[j].nodes[b];
                    const auto hx = static_cast<std::ptrdiff_t>(u.x) - static_cast<std::ptrdiff_t>(v.x);
                    const auto hy = static_cast<std::ptrdiff_t>(u.y) - static_cast<std::ptrdiff_t>(v.y);
                    entry += u.weight * v.weight * spectral.correlation(hx, hy);
                }
            }
            double pivot = entry;
            for (std::size_t k = 0; k < j; ++k) pivot -= lower[i*n+k] * lower[j*n+k];
            if (j == i) {
                if (!(pivot > 64. * std::numeric_limits<double>::epsilon() * entry))
                    throw std::invalid_argument("GRFS dependent or ill-conditioned hard observations");
                lower[i*n+i] = std::sqrt(pivot);
            } else {
                lower[i*n+j] = pivot / lower[j*n+j];
            }
        }
    }
    return lower;
}

inline void solve_constraints(const std::vector<double>& lower, std::vector<double>& rhs) {
    const auto n = rhs.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) rhs[i] -= lower[i*n+j] * rhs[j];
        rhs[i] /= lower[i*n+i];
    }
    // Column update in L^T, reading one contiguous row of L at a time.
    for (std::size_t i = n; i-- > 0;) {
        rhs[i] /= lower[i*n+i];
        for (std::size_t j = 0; j < i; ++j) rhs[j] -= lower[i*n+j] * rhs[i];
    }
}

inline std::vector<double> validate_trend(const Grid& grid, std::vector<double> trend) {
    require(trend.empty() || trend.size() == grid.nx * grid.ny, "GRFS trend size differs from grid size");
    for (double z : trend) require(std::isfinite(z), "GRFS nonfinite trend elevation");
    return trend;
}

}  // namespace detail

class Simulator {
    Grid grid_;
    std::vector<double> trend_;
    std::vector<detail::Constraint> points_;
    detail::Model model_;
    detail::Spectral spectral_;
    std::vector<double> lower_, targets_;
    double scale_, mean_;
    std::size_t rx_ = 0, ry_ = 0, width_ = 0;
    std::vector<double> stencil_;

    void prepare_correction() {
        if (points_.empty() || !model_.compact() || spectral_.clipped()) return;
        rx_ = (std::min)(model_.radius_x(), grid_.nx - 1);
        ry_ = (std::min)(model_.radius_y(), grid_.ny - 1);
        width_ = 2*rx_ + 1;
        const auto height = 2*ry_ + 1;
        std::size_t sources = 0;
        for (const auto& point : points_) sources += point.count;
        const double m = static_cast<double>(spectral_.nx() * spectral_.ny());
        const double area = static_cast<double>(width_) * static_cast<double>(height);
        // A dense neighborhood or many controls favor spectral convolution.
        if (static_cast<double>(sources) * area > m * (1. + std::log2(m)) ||
            width_ > stencil_.max_size() / height || area > m) return;
        stencil_.resize(width_ * height);
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width_; ++x) {
                const auto hx = static_cast<std::ptrdiff_t>(x) - static_cast<std::ptrdiff_t>(rx_);
                const auto hy = static_cast<std::ptrdiff_t>(y) - static_cast<std::ptrdiff_t>(ry_);
                stencil_[y*width_+x] = spectral_.correlation(hx, hy);
            }
        }
    }

    void correct(std::vector<double>& field, const std::vector<double>& coefficients) const {
        if (stencil_.empty()) {
            std::vector<double> impulses(field.size());
            for (std::size_t i = 0; i < points_.size(); ++i) {
                for (std::size_t s = 0; s < points_[i].count; ++s) {
                    const auto& node = points_[i].nodes[s];
                    impulses[node.index] += node.weight * coefficients[i];
                }
            }
            const auto correction = spectral_.convolve(impulses);
            for (std::size_t i = 0; i < field.size(); ++i) field[i] += correction[i];
            return;
        }
        // Apply C H^T locally wherever a source has nonzero covariance.
        // The coefficients still come from one global kriging system.
        for (std::size_t i = 0; i < points_.size(); ++i) {
            for (std::size_t s = 0; s < points_[i].count; ++s) {
                const auto& node = points_[i].nodes[s];
                const double weight = coefficients[i] * node.weight;
                const auto x0 = node.x > rx_ ? node.x - rx_ : 0;
                const auto y0 = node.y > ry_ ? node.y - ry_ : 0;
                const auto x1 = (std::min)(node.x + rx_, grid_.nx - 1);
                const auto y1 = (std::min)(node.y + ry_, grid_.ny - 1);
                for (std::size_t y = y0; y <= y1; ++y) {
                    const auto offset = (y + ry_ - node.y)*width_ + (x0 + rx_ - node.x);
                    for (std::size_t x = x0; x <= x1; ++x)
                        field[y*grid_.nx+x] += weight * stencil_[offset + (x-x0)];
                }
            }
        }
    }

public:
    Simulator(Grid grid, const Covariance& covariance, const std::vector<Point>& points = {},
              std::vector<double> trend = {}, Options options = {})
        : grid_(detail::validate_grid(grid, options)), trend_(detail::validate_trend(grid_, std::move(trend))),
          points_(detail::prepare_points(grid_, points, options.max_control_points)), model_(grid_, covariance),
          spectral_(grid_, model_, options), lower_(detail::factor_constraints(points_, spectral_)),
          scale_(std::sqrt(model_.sill())), mean_(options.mean.value_or(0.)) {
        if (trend_.empty() && !options.mean && !points_.empty()) {
            // Scale first, so an average of large finite observations remains finite.
            double largest = 0.;
            for (const auto& p : points_) largest = (std::max)(largest, std::abs(p.target));
            if (largest > 0.) {
                double sum = 0.;
                for (const auto& p : points_) sum += p.target / largest;
                mean_ = (sum / static_cast<double>(points_.size())) * largest;
            }
        }
        targets_.reserve(points_.size());
        for (const auto& p : points_) {
            const double local_mean = trend_.empty() ? mean_ : p.evaluate(trend_);
            const double residual = (p.target - local_mean) / scale_;
            if (!std::isfinite(residual)) throw std::overflow_error("GRFS standardized elevation overflow");
            targets_.push_back(residual);
        }
        prepare_correction();
    }

    PreparationInfo info() const {
        return {spectral_.nx(), spectral_.ny(), points_.size(), !stencil_.empty()};
    }

    std::vector<double> generate(std::uint64_t seed) const {
        auto field = spectral_.sample(seed);
        if (!points_.empty()) {
            auto rhs = targets_;
            for (std::size_t i = 0; i < rhs.size(); ++i) rhs[i] -= points_[i].evaluate(field);
            detail::solve_constraints(lower_, rhs);
            correct(field, rhs);
        }
        for (std::size_t i = 0; i < field.size(); ++i) {
            field[i] = field[i] * scale_ + (trend_.empty() ? mean_ : trend_[i]);
            if (!std::isfinite(field[i])) throw std::overflow_error("GRFS elevation overflow");
        }
        // Enforce a numerical contract even for nearly dependent constraints.
        for (const auto& p : points_) {
            double magnitude = (std::max)({1., scale_, std::abs(p.target)});
            for (std::size_t s = 0; s < p.count; ++s)
                magnitude = (std::max)(magnitude, std::abs(field[p.nodes[s].index]));
            if (!(std::abs(p.evaluate(field) - p.target) <= 1e-9 * magnitude))
                throw std::runtime_error("GRFS hard observation residual exceeds numerical tolerance");
        }
        return field;
    }
};

}  // namespace surface_grfs
#endif  // GAUSSIAN_SURFACE_SURFACE_GRFS_HPP
