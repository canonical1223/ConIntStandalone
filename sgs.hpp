#ifndef GAUSSIAN_SURFACE_SGS_HPP
#define GAUSSIAN_SURFACE_SGS_HPP

// Independent sequential Gaussian simulation on a Cartesian 3-D grid, C++17.
// A random path visits every active, unknown cell. Each draw conditions on
// exact cell observations and earlier draws in its region and search ellipsoid.
// The optional empirical transform maps the property to/from Gaussian scores.
// Example in a function body (after including this header):
//   sgs::Grid grid{40,30,8,{0,0,0},{10,10,1}};
//   sgs::Covariance cov{0.1,{{sgs::Kernel::spherical,0.9,{{80,35,5},30,0,0}}}};
//   sgs::Options opt; opt.seed=42; opt.search={{100,60,8},30,0,0};
//   auto table=sgs::NormalScores::fit({0.08,0.12,0.16,0.20,0.24});
//   auto porosity=sgs::simulate(grid,cov,opt,{{grid.index(2,3,0),0.16}}, {}, &table);
// Use a representative distribution for fit(); use different seeds for an ensemble.
// See docs/sgs.md for methodology. SLB describes the 2024.1 retirement of GSLIB
// under "Property Modeling" in the cumulative Petrel 2024.5 release notes:
// https://www.software.slb.com/software-news/support-news/petrel/petrel-studio-2024-5

/*
 * Последовательное гауссовское моделирование непрерывного свойства
 *
 * После осреднения каротажа до масштаба ячейки задайте по одному точному значению
 * свойства на ячейку. Для пористости опорное распределение берут из представительной
 * совокупности осреднённых значений той же фации и зоны. NormalScores::fit группирует
 * совпадающие значения, находит середину вероятностной массы группы и применяет
 * обратную функцию распределения стандартной нормали. Веса задают вклад в это
 * распределение. Для моделирования нормальных скоров обычно берут mean=0 и sill=1.
 * Если данные уже в нужном гауссовском пространстве, параметр transform опускают.
 *
 * Структуры ковариации суммируются. Для лага h в локальных осях структуры
 * r² = (h1/a)² + (h2/b)² + (h3/c)². При h≠0 каждая структура даёт вклад
 * partial_sill * rho(r). Сферическая rho=(1-r)²(1+r/2) при r<1, далее 0.
 * Экспоненциальная rho=exp(-ln(20)r), гауссовская rho=exp(-ln(20)r²).
 * C(0) равно сумме partial_sill и nugget. Полувариограмма gamma(h)=C(0)-C(h).
 * Эллипсоид поиска задаётся отдельно. Ближайшие соседи выбираются по расстоянию
 * в его нормированных координатах, на равном вычисленном расстоянии — по индексу.
 *
 * На каждом шаге d объединяет значения ближайших наблюдений и уже заполненных
 * ячеек своего региона. Пусть Kij=C(ui-uj), ci=C(ui-u). В простом кригинге
 * решается K w=c. Условное среднее m=mean+w'(d-mean*1), дисперсия v=C(0)-w'c.
 * В обычном кригинге K w+lambda*1=c и 1'w=1, тогда m=w'd, v=C(0)-w'c-lambda.
 * Следующее значение равно m+sqrt(v)*epsilon при независимом epsilon~N(0,1).
 * При пустой окрестности используются options.mean и полная априорная дисперсия.
 * Холецкий применяется к нормированной K. Потеря положительной определённости
 * порождает исключение. Ограничение соседей приближает полное обусловливание.
 *
 * Регионы с разными ID изолированы. Для разных гистограмм или вариограмм выполняют
 * отдельный вызов на каждый домен. Обратное преобразование ограничено минимумом
 * и максимумом таблицы, поэтому на границах возможны скопления значений. Для QC
 * проверьте точное воспроизведение скважинных ячеек, гистограммы по доменам,
 * направленные вариограммы скоров и разброс в ансамбле с разными seed. Идентичная
 * сборка с теми же параметрами и seed воспроизводит реализацию. Операции shuffle
 * и normal_distribution зависят от стандартной библиотеки C++.
 *
 * Эта реализация задаёт собственные численные соглашения для декартовой сетки.
 * В Petrel скважинные кривые сначала осредняют в ячейках (Scale up well logs).
 * В Petrophysical modeling выбирают свойство, зону/фацию, метод Sequential Gaussian
 * simulation, вариограмму и распределение. В практикуме KAU для пористости Ness 2
 * использованы экспоненциальная модель, дальности 3500/1500/10 и азимут 25 градусов.
 * Это настройки одного упражнения. В работе по Montney пористость откалибровали
 * по керну, осреднили в ячейках и смоделировали SGS на сетке 75 x 75 x 1.6 м.
 * Для проницаемости авторы применили установленную зависимость от пористости.
 * Историческая программа SGSIM описана в GSLIB, 2-е изд., раздел V.7.2. Сборка
 * Petrel, параметры поиска, геометрия и хвосты распределения важны для сравнения.
 * Численный эталон этого файла — условный гауссов закон на малой сетке.
 * KAU, практикум по Petrel, раздел 11.2
 * https://www.kau.edu.sa/Files/0002053/Subjects/Week11.pdf#page=3
 * Clarkson и др., Montney, раздел 2.1.1
 * https://www.mdpi.com/1996-1073/14/19/6007
 * Deutsch и Journel, историческое описание GSLIB
 * https://claytonvdeutsch.com/wp-content/uploads/2019/03/GSLIB-Book-Second-Edition.pdf
 * SLB, Petrel Petrophysical Modeling
 * https://www.slb.com/products-and-services/delivering-digital-at-scale/software/petrel-subsurface-software/petrel/petrel-geology-and-modeling/petrel-petrophysical-modeling
 * R. Ortiz, C. V. Deutsch, Multivariate Gaussian Distribution (2022), раздел SGS
 * https://geostatisticslessons.com/pdfs/multigaussian.pdf
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sgs {

struct Vec3 {
    double x = 0, y = 0, z = 0;
};

namespace detail {
inline void require(bool ok, const char *message) {
    if (!ok)
        throw std::invalid_argument(message);
}
inline bool finite(Vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
inline double component(Vec3 v, unsigned axis) { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }
inline Vec3 difference(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline std::size_t product(std::size_t a, std::size_t b) {
    if (a && b > (std::numeric_limits<std::size_t>::max)() / a)
        throw std::length_error("SGS size exceeds size_t");
    return a * b;
}
} // namespace detail

struct Grid {
    std::size_t nx = 1, ny = 1, nz = 1;
    Vec3 origin{}; // Centre of (0,0,0); Cartesian X east, Y north, Z up.
    Vec3 spacing{1, 1, 1};

    std::size_t size() const {
        detail::require(nx && ny && nz, "SGS grid dimensions must be positive");
        return detail::product(detail::product(nx, ny), nz);
    }
    void validate() const {
        (void)size();
        detail::require(detail::finite(origin) && detail::finite(spacing) && spacing.x > 0 &&
                            spacing.y > 0 && spacing.z > 0,
                        "SGS grid needs finite origin and positive finite spacing");
        auto check_axis = [](std::size_t count, double first, double step) {
            const double extent = double(count - 1) * step;
            const double last = std::fma(double(count - 1), step, first);
            detail::require(std::isfinite(extent) && std::isfinite(last), "SGS grid coordinates overflow");
            if (count == 1)
                return;
            const double magnitude = (std::max)(std::fabs(first), std::fabs(last));
            const double resolution = magnitude - std::nextafter(magnitude, 0.0);
            detail::require(last > first && step >= resolution && double(count - 1) < 0x1p52,
                            "SGS grid spacing is unresolved at this coordinate magnitude");
        };
        check_axis(nx, origin.x, spacing.x);
        check_axis(ny, origin.y, spacing.y);
        check_axis(nz, origin.z, spacing.z);
    }
    std::size_t index(std::size_t i, std::size_t j, std::size_t k) const {
        (void)size();
        if (i >= nx || j >= ny || k >= nz)
            throw std::out_of_range("SGS grid index");
        return i + nx * (j + ny * k);
    }
    // Explicit assignment to the containing voxel. Lower faces are inclusive,
    // upper faces exclusive. Upscale multiple log samples externally first.
    std::size_t cell_at(Vec3 point) const {
        validate();
        auto cell = [](double x, double first, double step, std::size_t count) {
            const double coordinate = (x - first) / step;
            double i = std::floor(coordinate);
            if (coordinate - i >= 0.5)
                ++i;
            if (!(i >= 0 && i < double(count)))
                throw std::out_of_range("SGS point outside grid");
            return static_cast<std::size_t>(i);
        };
        return index(cell(point.x, origin.x, spacing.x, nx), cell(point.y, origin.y, spacing.y, ny),
                     cell(point.z, origin.z, spacing.z, nz));
    }
};

// Semiaxes in the same length units as the grid. Azimuth rotates from north
// towards east. Positive dip tilts the first axis down. Positive roll rotates
// the second axis (initially horizontal, left of the first) towards the third
// axis (initially up when dip is zero), around the first by the right-hand rule.
struct Ellipsoid {
    Vec3 ranges{1, 1, 1};
    double azimuth_deg = 0, dip_deg = 0, roll_deg = 0;
};
enum class Kernel { spherical, exponential, gaussian };
struct Structure {
    Kernel kernel = Kernel::spherical;
    double partial_sill = 1;
    Ellipsoid geometry{};
};
struct Covariance {
    double nugget = 0; // Independent cell-scale variance, including at observed cells.
    std::vector<Structure> structures{Structure{}};
    double sill() const;
    double value(Vec3 lag) const;
};
enum class Kriging { simple, ordinary };
struct Options {
    std::uint64_t seed = 0;
    Kriging kriging = Kriging::simple;
    double mean = 0; // Known constant for SK; unconditional fallback for OK.
    Ellipsoid search{{8, 8, 8}, 0, 0, 0};
    std::size_t max_neighbors = 16;
};
struct CellData {
    std::size_t cell;
    double value;
};

namespace detail {
class Frame {
    Vec3 a_, b_, c_, ranges_;

public:
    explicit Frame(const Ellipsoid &e) : ranges_(e.ranges) {
        require(finite(ranges_) && ranges_.x > 0 && ranges_.y > 0 && ranges_.z > 0 &&
                    std::isfinite(e.azimuth_deg) && std::isfinite(e.dip_deg) && std::isfinite(e.roll_deg),
                "SGS invalid ellipsoid ranges or angles");
        auto radians = [](double deg) { return std::remainder(deg, 360.0) * 0.01745329251994329577; };
        const double az = radians(e.azimuth_deg), dip = radians(e.dip_deg), roll = radians(e.roll_deg);
        const double sa = std::sin(az), ca = std::cos(az), sd = std::sin(dip), cd = std::cos(dip);
        const double sr = std::sin(roll), cr = std::cos(roll);
        a_ = {sa * cd, ca * cd, -sd};
        b_ = {-ca * cr + sa * sd * sr, sa * cr + ca * sd * sr, cd * sr};
        c_ = {ca * sr + sa * sd * cr, -sa * sr + ca * sd * cr, cd * cr};
    }
    Vec3 normalize(Vec3 lag) const {
        return {dot(a_, lag) / ranges_.x, dot(b_, lag) / ranges_.y, dot(c_, lag) / ranges_.z};
    }
};

class Model {
    struct Term {
        Kernel kernel;
        double fraction;
        Frame frame;
    };
    std::vector<Term> terms_;

public:
    double sill;
    explicit Model(const Covariance &model) : sill(model.nugget) {
        require(std::isfinite(sill) && sill >= 0, "SGS invalid nugget");
        for (const Structure &s : model.structures) {
            require(std::isfinite(s.partial_sill) && s.partial_sill >= 0, "SGS invalid partial sill");
            require(s.kernel == Kernel::spherical || s.kernel == Kernel::gaussian ||
                        s.kernel == Kernel::exponential,
                    "SGS unknown kernel");
            terms_.push_back({s.kernel, s.partial_sill, Frame(s.geometry)});
            sill += s.partial_sill;
        }
        require(std::isfinite(sill) && sill > 0, "SGS total sill must be positive and finite");
        for (auto &term : terms_)
            term.fraction /= sill;
    }
    // Correlation is precisely 0.05 at unit normalized distance for the two
    // asymptotic models. Spherical correlation reaches zero at unit distance.
    double correlation(Vec3 lag) const {
        if (lag.x == 0 && lag.y == 0 && lag.z == 0)
            return 1;
        double result = 0;
        constexpr double log20 = 2.9957322735539909934;
        for (const auto &term : terms_) {
            if (term.fraction == 0)
                continue;
            const Vec3 u = term.frame.normalize(lag);
            const double r2 = dot(u, u);
            if (std::isnan(r2))
                throw std::overflow_error("SGS covariance coordinate overflow");
            double rho;
            if (term.kernel == Kernel::gaussian)
                rho = std::exp(-log20 * r2);
            else if (term.kernel == Kernel::exponential)
                rho = std::exp(-log20 * std::sqrt(r2));
            else {
                if (r2 >= 1)
                    continue;
                const double r = std::sqrt(r2);
                rho = (1 - r) * (1 - r) * (1 + 0.5 * r);
            }
            result += term.fraction * rho;
        }
        return result;
    }
};
} // namespace detail

inline double Covariance::sill() const { return detail::Model(*this).sill; }
inline double Covariance::value(Vec3 lag) const {
    detail::require(detail::finite(lag), "SGS covariance lag must be finite");
    const detail::Model prepared(*this);
    return prepared.sill * prepared.correlation(lag);
}

// Weighted empirical distribution. Equal values share their group's midrank.
// Interpolate linearly in Gaussian-score coordinates and clamp both tails to
// the observed endpoints. All observations for a simulation must be supported
// by its table. Use distinct tables/calls for domains with different marginals.
class NormalScores {
    std::vector<double> values_, scores_;
    NormalScores() = default;

    static double lower_quantile(double p) {
        if (!(p > 0))
            throw std::range_error("SGS normal-score probability underflow");
        if (p >= 0.5)
            return 0;
        double lo = -40, hi = 0;
        for (unsigned i = 0; i < 64; ++i) {
            const double mid = (lo + hi) * 0.5;
            if (0.5 * std::erfc(-mid * 0.70710678118654752440) < p)
                lo = mid;
            else
                hi = mid;
        }
        return (lo + hi) * 0.5;
    }
    static double interpolate(double x, const std::vector<double> &from, const std::vector<double> &to) {
        detail::require(std::isfinite(x), "SGS transform input must be finite");
        if (from.empty())
            throw std::logic_error("SGS empty normal-score table");
        if (x <= from.front())
            return to.front();
        if (x >= from.back())
            return to.back();
        const auto it = std::lower_bound(from.begin(), from.end(), x);
        const auto j = static_cast<std::size_t>(it - from.begin());
        if (*it == x)
            return to[j];
        const double a = from[j - 1], b = from[j], width = b - a;
        const double t = std::isfinite(width) ? (x - a) / width : (x * 0.5 - a * 0.5) / (b * 0.5 - a * 0.5);
        return (1 - t) * to[j - 1] + t * to[j];
    }

public:
    static NormalScores fit(const std::vector<double> &values, const std::vector<double> &weights = {}) {
        detail::require(!values.empty() && (weights.empty() || values.size() == weights.size()),
                        "SGS normal-score fit needs values and matching optional weights");
        std::vector<std::pair<double, double>> rows;
        double max_weight = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double w = weights.empty() ? 1 : weights[i];
            detail::require(std::isfinite(values[i]) && std::isfinite(w) && w >= 0,
                            "SGS nonfinite value or invalid normal-score weight");
            if (w > 0) {
                rows.emplace_back(values[i], w);
                max_weight = (std::max)(max_weight, w);
            }
        }
        std::sort(rows.begin(), rows.end());
        NormalScores table;
        std::vector<long double> mass;
        for (const auto &row : rows) {
            if (mass.empty() || row.first != table.values_.back()) {
                table.values_.push_back(row.first);
                mass.push_back(0);
            }
            mass.back() += static_cast<long double>(row.second) / max_weight;
        }
        detail::require(mass.size() >= 2, "SGS normal-score fit needs two distinct positive-weight values");
        // Compute each tail from its nearer end; 1-CDF loses small upper-tail weights.
        std::vector<long double> upper(mass.size());
        long double total = 0;
        for (std::size_t i = mass.size(); i-- > 0;) {
            upper[i] = total + 0.5L * mass[i];
            total += mass[i];
        }
        long double prefix = 0;
        for (std::size_t i = 0; i < mass.size(); ++i) {
            const long double lower = prefix + 0.5L * mass[i];
            prefix += mass[i];
            const double q = lower_quantile(static_cast<double>((std::min)(lower, upper[i]) / total));
            const double score = lower <= upper[i] ? q : -q;
            if (i && score <= table.scores_.back())
                throw std::range_error("SGS normal-score weights exceed numeric resolution");
            table.scores_.push_back(score);
        }
        return table;
    }
    double forward(double value) const { return interpolate(value, values_, scores_); }
    double inverse(double score) const { return interpolate(score, scores_, values_); }
    bool contains(double value) const {
        return !values_.empty() && value >= values_.front() && value <= values_.back();
    }
};

namespace detail {

// The tree is built once per region in dimensionless search coordinates.
// Its split planes prune by ellipsoidal distance; subtree counts prune cells
// that have not yet received any data. A capped heap keeps the nearest points.
struct KdNode {
    Vec3 position;
    std::size_t cell, informed_count = 0;
    unsigned axis = 0;
};
using Neighbor = std::pair<double, std::size_t>; // squared distance, then linear cell index
using Heap = std::priority_queue<Neighbor>;

class Neighborhood {
    std::vector<KdNode> nodes_;
    static std::size_t middle(std::size_t lo, std::size_t hi) { return lo + (hi - lo) / 2; }
    void build(std::size_t lo, std::size_t hi, std::vector<std::size_t> &tree_positions) {
        if (lo == hi)
            return;
        Vec3 low = nodes_[lo].position, high = low;
        for (std::size_t i = lo + 1; i < hi; ++i) {
            const Vec3 p = nodes_[i].position;
            low = {(std::min)(low.x, p.x), (std::min)(low.y, p.y), (std::min)(low.z, p.z)};
            high = {(std::max)(high.x, p.x), (std::max)(high.y, p.y), (std::max)(high.z, p.z)};
        }
        const Vec3 spread = difference(high, low);
        unsigned axis = spread.y > spread.x ? 1 : 0;
        if (spread.z > component(spread, axis))
            axis = 2;
        const std::size_t mid = middle(lo, hi);
        std::nth_element(nodes_.begin() + lo, nodes_.begin() + mid, nodes_.begin() + hi,
                         [axis](const KdNode &a, const KdNode &b) {
                             const double x = component(a.position, axis), y = component(b.position, axis);
                             return x == y ? a.cell < b.cell : x < y;
                         });
        nodes_[mid].axis = axis;
        tree_positions[nodes_[mid].cell] = mid;
        build(lo, mid, tree_positions);
        build(mid + 1, hi, tree_positions);
    }
    static double radius2(const Heap &heap, std::size_t cap) {
        return heap.size() == cap ? heap.top().first : 1 + 16 * std::numeric_limits<double>::epsilon();
    }
    void visit(std::size_t lo, std::size_t hi, Vec3 target, std::size_t cap,
               const std::vector<double> &known, Heap &heap) const {
        if (lo == hi)
            return;
        const std::size_t mid = middle(lo, hi);
        const auto &node = nodes_[mid];
        if (!node.informed_count)
            return;
        const Vec3 diff = difference(target, node.position);
        const double plane = component(diff, node.axis);
        if (std::isfinite(known[node.cell])) {
            const double dist = dot(diff, diff);
            const Neighbor candidate{dist, node.cell};
            if (dist <= radius2(heap, cap) && (heap.size() < cap || candidate < heap.top())) {
                if (heap.size() == cap)
                    heap.pop();
                heap.push(candidate);
            }
        }
        const bool left = plane <= 0;
        visit(left ? lo : mid + 1, left ? mid : hi, target, cap, known, heap);
        if (plane * plane <= radius2(heap, cap))
            visit(left ? mid + 1 : lo, left ? hi : mid, target, cap, known, heap);
    }

public:
    void add(Vec3 position, std::size_t cell) {
        require(finite(position), "SGS grid exceeds search-coordinate range");
        nodes_.push_back({position, cell});
    }
    void build(std::vector<std::size_t> &tree_positions) { build(0, nodes_.size(), tree_positions); }
    void mark(std::size_t slot) {
        std::size_t lo = 0, hi = nodes_.size();
        for (;;) {
            const std::size_t mid = middle(lo, hi);
            ++nodes_[mid].informed_count;
            if (mid == slot)
                return;
            if (slot < mid)
                hi = mid;
            else
                lo = mid + 1;
        }
    }
    void nearest(std::size_t slot, std::size_t cap, const std::vector<double> &known,
                 std::vector<std::size_t> &out) const {
        Heap heap;
        visit(0, nodes_.size(), nodes_[slot].position, cap, known, heap);
        out.resize(heap.size());
        for (std::size_t i = out.size(); i-- > 0;) {
            out[i] = heap.top().second;
            heap.pop();
        }
    }
};

inline Vec3 grid_lag(const Grid &g, std::size_t a, std::size_t b) {
    auto difference_index = [](std::size_t i, std::size_t j) {
        return i >= j ? static_cast<double>(i - j) : -static_cast<double>(j - i);
    };
    const double x = difference_index(a % g.nx, b % g.nx) * g.spacing.x;
    a /= g.nx;
    b /= g.nx;
    return {x, difference_index(a % g.ny, b % g.ny) * g.spacing.y,
            difference_index(a / g.ny, b / g.ny) * g.spacing.z};
}

struct Conditional {
    double mean, variance;
};
class KrigingWorkspace {
    std::vector<double> factor_, target_, residual_, ones_;

public:
    Conditional condition(const Grid &grid, const Model &model, const Options &opt, std::size_t cell,
                          const std::vector<std::size_t> &neighbors, const std::vector<double> &known) {
        const std::size_t n = neighbors.size();
        if (!n)
            return {opt.mean, model.sill};
        const bool ordinary = opt.kriging == Kriging::ordinary;
        factor_.resize(product(n, n));
        target_.resize(n);
        residual_.resize(n);
        if (ordinary)
            ones_.resize(n);
        // Factor K/sill = L*L'. Simultaneously whiten target correlations,
        // conditioning residuals and (for ordinary kriging) the vector of ones.
        // Center OK at a neighbor so the arbitrary unconditional mean does not
        // affect a nonempty neighborhood, even when its magnitude is very large.
        const double center = ordinary ? known[neighbors.front()] : opt.mean;
        for (std::size_t i = 0; i < n; ++i) {
            double target = model.correlation(grid_lag(grid, neighbors[i], cell));
            double residual = known[neighbors[i]] - center, one = 1;
            double *row = factor_.data() + i * n;
            for (std::size_t j = 0; j <= i; ++j) {
                double v = i == j ? 1 : model.correlation(grid_lag(grid, neighbors[i], neighbors[j]));
                const double *prev = factor_.data() + j * n;
                for (std::size_t k = 0; k < j; ++k)
                    v -= row[k] * prev[k];
                if (i == j) {
                    if (!(v > 0 && std::isfinite(v)))
                        throw std::runtime_error("SGS covariance matrix not numerically positive definite");
                    row[i] = std::sqrt(v);
                } else {
                    row[j] = v / prev[j];
                    target -= row[j] * target_[j];
                    residual -= row[j] * residual_[j];
                    if (ordinary)
                        one -= row[j] * ones_[j];
                }
            }
            target_[i] = target / row[i];
            residual_[i] = residual / row[i];
            if (ordinary)
                ones_[i] = one / row[i];
        }
        const auto dotn = [](const std::vector<double> &a, const std::vector<double> &b) {
            return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
        };
        const double squared = dotn(target_, target_);
        double mean = center + dotn(target_, residual_), variance = 1 - squared;
        if (ordinary) {
            const double delta = 1 - dotn(target_, ones_);
            const double norm = dotn(ones_, ones_);
            if (!(norm > 0 && std::isfinite(norm)))
                throw std::runtime_error("SGS unstable ordinary kriging");
            mean += (delta / norm) * dotn(ones_, residual_);
            variance += (delta / norm) * delta;
        }
        const double tolerance = 64 * std::numeric_limits<double>::epsilon() * double(n) * (1 + squared);
        if (!std::isfinite(variance) || variance < -tolerance)
            throw std::runtime_error("SGS invalid conditional variance; inspect covariance conditioning");
        variance = (std::max)(0.0, variance) * model.sill;
        if (!std::isfinite(mean) || !std::isfinite(variance))
            throw std::overflow_error("SGS conditional moments overflow");
        return {mean, variance};
    }
};
} // namespace detail

// One realization, x index fastest. Empty regions = single active domain.
// Negative region IDs are inactive (returned as NaN). All nonnegative IDs
// share options and covariance, but condition exclusively on their own data.
// Cell data are exact point-support constraints at grid centers; a conflicting
// duplicate or a datum in an inactive cell throws. If a table is supplied,
// data and results are in property units; covariance and mean are in score space.
inline std::vector<double> simulate(const Grid &grid, const Covariance &covariance,
                                    const Options &options = {}, const std::vector<CellData> &hard = {},
                                    const std::vector<int> &regions = {},
                                    const NormalScores *transform = nullptr) {
    grid.validate();
    const std::size_t count = grid.size();
    detail::require(regions.empty() || regions.size() == count, "SGS region array size mismatch");
    detail::require(std::isfinite(options.mean) && options.max_neighbors > 0 &&
                        (options.kriging == Kriging::simple || options.kriging == Kriging::ordinary),
                    "SGS invalid mean, kriging type or neighbor cap");
    const detail::Model model(covariance);
    const detail::Frame search(options.search);
    auto region = [&regions](std::size_t cell) { return regions.empty() ? 0 : regions[cell]; };
    std::vector<double> values(count, std::numeric_limits<double>::quiet_NaN());
    for (const CellData &datum : hard) {
        detail::require(datum.cell < count && region(datum.cell) >= 0,
                        "SGS hard cell is invalid or inactive");
        detail::require(std::isfinite(datum.value) && (!transform || transform->contains(datum.value)),
                        "SGS hard value must be finite and within transform support");
        double &existing = values[datum.cell];
        detail::require(std::isnan(existing) || existing == datum.value,
                        "SGS conflicting values in one cell");
        existing = datum.value;
    }
    std::map<int, detail::Neighborhood> domains;
    std::vector<std::size_t> tree_positions(count), path;
    for (std::size_t cell = 0; cell < count; ++cell) {
        const int id = region(cell);
        if (id < 0)
            continue;
        domains[id].add(search.normalize(detail::grid_lag(grid, cell, 0)), cell);
        if (std::isnan(values[cell]))
            path.push_back(cell);
        else if (transform)
            values[cell] = transform->forward(values[cell]);
    }
    for (auto &domain : domains)
        domain.second.build(tree_positions);
    for (std::size_t cell = 0; cell < count; ++cell) {
        if (std::isfinite(values[cell]))
            domains.at(region(cell)).mark(tree_positions[cell]);
    }
    std::mt19937_64 rng(options.seed);
    std::shuffle(path.begin(), path.end(), rng);
    std::normal_distribution<double> normal;
    std::vector<std::size_t> neighbors;
    detail::KrigingWorkspace solver;
    for (std::size_t cell : path) {
        auto &domain = domains.at(region(cell));
        domain.nearest(tree_positions[cell], options.max_neighbors, values, neighbors);
        const auto moments = solver.condition(grid, model, options, cell, neighbors, values);
        values[cell] = moments.mean + std::sqrt(moments.variance) * normal(rng);
        if (!std::isfinite(values[cell]))
            throw std::overflow_error("SGS simulated value overflow");
        domain.mark(tree_positions[cell]);
    }
    if (transform) {
        for (std::size_t cell : path)
            values[cell] = transform->inverse(values[cell]);
        for (const CellData &datum : hard)
            values[datum.cell] = datum.value;
    }
    return values;
}

} // namespace sgs
#endif // GAUSSIAN_SURFACE_SGS_HPP
