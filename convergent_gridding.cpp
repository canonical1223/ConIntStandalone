#include "convergent_gridding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace convergent {
namespace {

// v1: validation/canonicalization -> coarse-to-fine [Refine, Snap/Taylor,
// minimum-curvature PCG] -> exact control projection -> qreal check -> commit.
// Численные этапы, свободные границы, порядок суммирования и допуски сохранены.
// Все внутренние типы, геометрия разломов и решатели принадлежат этому модулю.

// Пересечение и любое касание разлома блокируют прямую видимость.
class FaultGeometry {
public:
    FaultGeometry(const Surface& surface, const std::vector<Fault>& faults);

    bool empty() const noexcept { return segments_.empty(); }
    bool onFault(double x, double y) const;
    bool visible(double ax, double ay, double bx, double by) const;

private:
    struct XY {
        long double x;
        long double y;
    };

    struct Box {
        long double minx;
        long double maxx;
        long double miny;
        long double maxy;
    };

    struct Segment {
        XY a;
        XY b;
        Box box;
    };

    // Preorder BVH: escape пропускает поддерево без стека и рекурсии.
    struct IndexNode {
        Box box;
        std::size_t begin;  // первый сегмент листа
        std::size_t count;  // 0 у внутреннего узла
        std::size_t escape;
    };

    XY normalize(double x, double y) const;
    static Box bounds(const XY& a, const XY& b);
    static bool overlaps(const Box& a, const Box& b);
    static bool contains(const Box& box, const XY& point);
    static int orientation(const XY& a, const XY& b, const XY& point);
    static bool intersects(const XY& a, const XY& b, const Box& box,
                           const Segment& segment);
    void buildIndex(std::size_t begin, std::size_t end);
    template<class BoxPredicate, class SegmentPredicate>
    bool anySegment(BoxPredicate boxMatches, SegmentPredicate segmentMatches) const;

    // Изотропное масштабирование сохраняет геометрию, перенос снижает округление.
    long double originX_{};
    long double originY_{};
    long double scale_{1};
    std::vector<Segment> segments_;
    std::vector<IndexNode> index_;
};

// Допуск в долях максимальной стороны поверхности.
constexpr long double geometryTolerance =
    64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon());

constexpr std::size_t linearScanLimit = 16;
constexpr std::size_t maximumLeafSegments = 8;

FaultGeometry::FaultGeometry(const Surface& surface,
                             const std::vector<Fault>& faults)
    : originX_(static_cast<long double>(surface.minx)),
      originY_(static_cast<long double>(surface.miny))
{
    const long double width = static_cast<long double>(surface.maxx) - originX_;
    const long double height = static_cast<long double>(surface.maxy) - originY_;
    if (!std::isfinite(originX_) || !std::isfinite(originY_) ||
        !std::isfinite(width) || !std::isfinite(height) ||
        !(width > 0) || !(height > 0)) {
        throw std::invalid_argument("Fault geometry requires finite positive surface extents");
    }
    scale_ = std::max(width, height);

    for (std::size_t faultIndex = 0; faultIndex < faults.size(); ++faultIndex) {
        const auto& fault = faults[faultIndex];
        const auto rejectFault = [faultIndex] {
            throw std::invalid_argument("Fault " + std::to_string(faultIndex) +
                                        " requires at least two distinct XY vertices");
        };
        if (fault.size() < 2) {
            rejectFault();
        }

        XY previous = normalize(fault.front().x, fault.front().y);
        const std::size_t initialSegmentCount = segments_.size();
        for (std::size_t vertex = 1; vertex < fault.size(); ++vertex) {
            const XY next = normalize(fault[vertex].x, fault[vertex].y);
            // Удаляются только точные повторы, без слияния близких вершин.
            if (previous.x == next.x && previous.y == next.y) {
                continue;
            }
            segments_.push_back({previous, next, bounds(previous, next)});
            previous = next;
        }
        if (segments_.size() == initialSegmentCount) {
            rejectFault();
        }
    }
    if (segments_.size() > linearScanLimit) {
        index_.reserve(segments_.size() / maximumLeafSegments);
        buildIndex(0, segments_.size());
    }
}

void FaultGeometry::buildIndex(std::size_t begin, std::size_t end)
{
    Box box = segments_[begin].box;
    for (std::size_t i = begin + 1; i < end; ++i) {
        const Box& child = segments_[i].box;
        box.minx = std::min(box.minx, child.minx);
        box.maxx = std::max(box.maxx, child.maxx);
        box.miny = std::min(box.miny, child.miny);
        box.maxy = std::max(box.maxy, child.maxy);
    }
    const std::size_t at = index_.size();
    index_.push_back({box, begin, end - begin, 0});
    if (end - begin > maximumLeafSegments) {
        // Медиана по большей стороне сохраняет логарифмическую глубину.
        const bool alongX = box.maxx - box.minx >= box.maxy - box.miny;
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(segments_.begin() + begin, segments_.begin() + middle,
                         segments_.begin() + end,
            [alongX](const Segment& a, const Segment& b) {
                return alongX ? a.box.minx + a.box.maxx < b.box.minx + b.box.maxx
                              : a.box.miny + a.box.maxy < b.box.miny + b.box.maxy;
            });
        index_[at].count = 0;
        buildIndex(begin, middle);
        buildIndex(middle, end);
    }
    index_[at].escape = index_.size();
}

FaultGeometry::XY FaultGeometry::normalize(double x, double y) const
{
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw std::invalid_argument("Fault geometry coordinates must be finite");
    }
    const XY result{
        (static_cast<long double>(x) - originX_) / scale_,
        (static_cast<long double>(y) - originY_) / scale_
    };
    // Внешние вершины разрешены; предел защищает детерминант от переполнения.
    const long double safeMagnitude =
        std::sqrt(std::numeric_limits<long double>::max()) / 32.0L;
    if (!std::isfinite(result.x) || !std::isfinite(result.y) ||
        std::abs(result.x) > safeMagnitude ||
        std::abs(result.y) > safeMagnitude) {
        throw std::invalid_argument("Fault geometry coordinates exceed the safe normalized range");
    }
    return result;
}

FaultGeometry::Box FaultGeometry::bounds(const XY& a, const XY& b)
{
    return {std::min(a.x, b.x), std::max(a.x, b.x),
            std::min(a.y, b.y), std::max(a.y, b.y)};
}

bool FaultGeometry::overlaps(const Box& a, const Box& b)
{
    return !(a.maxx + geometryTolerance < b.minx ||
             b.maxx + geometryTolerance < a.minx ||
             a.maxy + geometryTolerance < b.miny ||
             b.maxy + geometryTolerance < a.miny);
}

bool FaultGeometry::contains(const Box& box, const XY& point)
{
    return point.x >= box.minx - geometryTolerance &&
           point.x <= box.maxx + geometryTolerance &&
           point.y >= box.miny - geometryTolerance &&
           point.y <= box.maxy + geometryTolerance;
}

int FaultGeometry::orientation(const XY& a, const XY& b, const XY& point)
{
    // Порог сочетает геометрический допуск и погрешность вычитания произведений.
    const long double dx = b.x - a.x;
    const long double dy = b.y - a.y;
    const long double term1 = dx * (point.y - a.y);
    const long double term2 = dy * (point.x - a.x);
    const long double determinant = term1 - term2;
    const long double roundoff =
        32.0L * std::numeric_limits<long double>::epsilon() *
        (std::abs(term1) + std::abs(term2));
    const long double threshold =
        geometryTolerance * std::max(std::abs(dx), std::abs(dy)) + roundoff;
    if (determinant > threshold) {
        return 1;
    }
    if (determinant < -threshold) {
        return -1;
    }
    return 0;
}

bool FaultGeometry::intersects(const XY& a, const XY& b, const Box& box,
                               const Segment& segment)
{
    if (!overlaps(box, segment.box)) {
        return false;
    }
    const int abA = orientation(a, b, segment.a);
    const int abB = orientation(a, b, segment.b);
    const int faultA = orientation(segment.a, segment.b, a);
    const int faultB = orientation(segment.a, segment.b, b);

    // Касания включают общие концы, наложение и вырожденный запрос a == b.
    if ((abA == 0 && contains(box, segment.a)) ||
        (abB == 0 && contains(box, segment.b)) ||
        (faultA == 0 && contains(segment.box, a)) ||
        (faultB == 0 && contains(segment.box, b))) {
        return true;
    }
    return abA * abB < 0 && faultA * faultB < 0;
}

template<class BoxPredicate, class SegmentPredicate>
bool FaultGeometry::anySegment(BoxPredicate boxMatches,
                               SegmentPredicate segmentMatches) const
{
    if (!index_.empty()) {
        std::size_t node = 0;
        while (node < index_.size()) {
            const IndexNode& candidate = index_[node];
            if (!boxMatches(candidate.box)) {
                node = candidate.escape;
                continue;
            }
            for (std::size_t i = candidate.begin; i < candidate.begin + candidate.count; ++i) {
                if (segmentMatches(segments_[i])) return true;
            }
            ++node;
        }
        return false;
    }
    for (const auto& segment : segments_) {
        if (segmentMatches(segment)) return true;
    }
    return false;
}

bool FaultGeometry::onFault(double x, double y) const
{
    const XY point = normalize(x, y);
    return anySegment(
        [&](const Box& box) { return contains(box, point); },
        [&](const Segment& segment) {
            return contains(segment.box, point) &&
                   orientation(segment.a, segment.b, point) == 0;
        });
}

bool FaultGeometry::visible(double ax, double ay, double bx, double by) const
{
    const XY a = normalize(ax, ay);
    const XY b = normalize(bx, by);
    const Box queryBounds = bounds(a, b);
    return !anySegment(
        [&](const Box& box) { return overlaps(queryBounds, box); },
        [&](const Segment& segment) { return intersects(a, b, queryBounds, segment); });
}

using Scalar = double;
struct CanonicalPoint {
    Scalar x{};
    Scalar y{};
    Scalar value{};
    Scalar weight{};
};
struct Grid {
    std::size_t nx{};
    std::size_t ny{};
    Scalar minx{};
    Scalar maxx{};
    Scalar miny{};
    Scalar maxy{};
    std::vector<Scalar> values;
    const FaultGeometry* faults{};

    std::size_t index(std::size_t ix, std::size_t iy) const noexcept
    {
        return iy * nx + ix;
    }

    Scalar dx() const noexcept
    {
        return (maxx - minx) / static_cast<Scalar>(nx - 1);
    }

    Scalar dy() const noexcept
    {
        return (maxy - miny) / static_cast<Scalar>(ny - 1);
    }

    Scalar x(std::size_t ix) const noexcept
    {
        return ix + 1 == nx ? maxx : minx + static_cast<Scalar>(ix) * dx();
    }

    Scalar y(std::size_t iy) const noexcept
    {
        return iy + 1 == ny ? maxy : miny + static_cast<Scalar>(iy) * dy();
    }
};
struct Derivatives {
    Scalar gx{};
    Scalar gy{};
    Scalar gxx{};
    Scalar gxy{};
    Scalar gyy{};
};
struct StencilTerm {
    std::size_t index{};
    Scalar coefficient{};
};
struct PointConstraint {
    std::array<StencilTerm, 4> terms{};
    std::size_t count{};
    Scalar value{};
    Scalar weight{};
};
struct SolverResult {
    std::vector<Scalar> values;
    std::size_t iterations{};
    Scalar relativeResidual{};
    bool converged{};
};
struct ProjectionResult {
    std::size_t iterations{};
    Scalar maxError{};
    bool converged{};
};

inline bool finite(Scalar value) noexcept
{
    return std::isfinite(value);
}
inline std::size_t checkedNodeCount(std::size_t nx, std::size_t ny)
{
    if (nx == 0 || ny > std::numeric_limits<std::size_t>::max() / nx) {
        throw std::invalid_argument("surface dimensions overflow size_t");
    }
    return nx * ny;
}

inline bool visibleNodes(const Grid& grid, std::size_t a, std::size_t b)
{
    return !grid.faults || grid.faults->visible(
        grid.x(a % grid.nx), grid.y(a / grid.nx),
        grid.x(b % grid.nx), grid.y(b / grid.nx));
}
inline Scalar constraintValue(const PointConstraint& constraint,
                       const std::vector<Scalar>& values)
{
    Scalar result = 0;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        result += constraint.terms[i].coefficient * values[constraint.terms[i].index];
    }
    return result;
}
inline Grid gridGeometry(const Grid& source)
{
    Grid grid;
    grid.nx = source.nx;
    grid.ny = source.ny;
    grid.minx = source.minx;
    grid.maxx = source.maxx;
    grid.miny = source.miny;
    grid.maxy = source.maxy;
    grid.faults = source.faults;
    return grid;
}

// Удаляем целую строку B при пересечении разлома любой парой ее узлов.
// Это сохраняет симметрию и положительную полуопределенность B^T*B.
std::vector<std::uint8_t> faultCurvatureMask(const Grid& grid)
{
    if (!grid.faults) return {};
    std::vector<std::uint8_t> mask(grid.values.size(), 0);
    const auto allowed = [&](std::initializer_list<std::size_t> nodes) {
        for (auto a = nodes.begin(); a != nodes.end(); ++a)
            for (auto b = a + 1; b != nodes.end(); ++b)
                if (!visibleNodes(grid, *a, *b)) return false;
        return true;
    };
    for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
            const auto center = grid.index(i, j);
            if (i > 0 && i + 1 < grid.nx && allowed({grid.index(i - 1, j), center, grid.index(i + 1, j)}))
                mask[center] |= 1;
            if (j > 0 && j + 1 < grid.ny && allowed({grid.index(i, j - 1), center, grid.index(i, j + 1)}))
                mask[center] |= 2;
            if (i > 0 && i + 1 < grid.nx && j > 0 && j + 1 < grid.ny
                && allowed({grid.index(i + 1, j + 1), grid.index(i + 1, j - 1),
                            grid.index(i - 1, j + 1), grid.index(i - 1, j - 1)}))
                mask[center] |= 4;
        }
    }
    return mask;
}

// Общий обход гарантирует одинаковые граничные строки и маску у K и diag(K).
// Порядок накопления сохранен: X, Y, смешанная производная; внутри — по строкам.
template<class Axial, class Mixed>
void visitCurvatureStencils(const Grid& grid, const std::vector<std::uint8_t>* mask,
                            const Axial& axial, const Mixed& mixed)
{
    const Scalar ax = grid.dy() / grid.dx();
    const Scalar ay = grid.dx() / grid.dy();
    if (grid.nx >= 3) {
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 1)) continue;
                axial(grid.index(ix - 1, iy), grid.index(ix, iy),
                      grid.index(ix + 1, iy), ax);
            }
        }
    }
    if (grid.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 2)) continue;
                axial(grid.index(ix, iy - 1), grid.index(ix, iy),
                      grid.index(ix, iy + 1), ay);
            }
        }
    }
    if (grid.nx >= 3 && grid.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < grid.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < grid.nx; ++ix) {
                if (mask && !((*mask)[grid.index(ix, iy)] & 4)) continue;
                mixed(grid.index(ix + 1, iy + 1), grid.index(ix + 1, iy - 1),
                      grid.index(ix - 1, iy + 1), grid.index(ix - 1, iy - 1));
            }
        }
    }
}

// Накапливает K*input, K=smoothness*B^T*B, не очищая output.
// Scatter последовательный: соседние строки пишут в общие узлы.
void applyCurvature(
    const Grid& grid,
    const std::vector<Scalar>& input,
    std::vector<Scalar>& output,
    Scalar smoothness,
    const std::vector<std::uint8_t>* mask = nullptr)
{
    if (smoothness == 0) return;
    visitCurvatureStencils(grid, mask,
        [&](std::size_t a, std::size_t b, std::size_t c, Scalar scale) {
            const Scalar d = scale * (input[a] - Scalar(2) * input[b] + input[c]);
            const Scalar weighted = smoothness * scale * d;
            output[a] += weighted;
            output[b] -= Scalar(2) * weighted;
            output[c] += weighted;
        },
        [&](std::size_t pp, std::size_t pm, std::size_t mp, std::size_t mm) {
            constexpr Scalar c = Scalar(0.25);
            const Scalar d = c * (input[pp] - input[pm] - input[mp] + input[mm]);
            const Scalar weighted = Scalar(2) * smoothness * c * d;
            output[pp] += weighted;
            output[pm] -= weighted;
            output[mp] -= weighted;
            output[mm] += weighted;
        });
}

// Точная диагональ K для предобуславливателя Якоби.
std::vector<Scalar> curvatureDiagonal(const Grid& grid, Scalar smoothness,
                                     const std::vector<std::uint8_t>* mask = nullptr)
{
    std::vector<Scalar> diagonal(grid.values.size(), Scalar(0));
    if (smoothness == 0) return diagonal;
    visitCurvatureStencils(grid, mask,
        [&](std::size_t a, std::size_t b, std::size_t c, Scalar scale) {
            const Scalar factor = smoothness * scale * scale;
            diagonal[a] += factor;
            diagonal[b] += Scalar(4) * factor;
            diagonal[c] += factor;
        },
        [&](std::size_t pp, std::size_t pm, std::size_t mp, std::size_t mm) {
            constexpr Scalar coefficientSquared = Scalar(1) / Scalar(16);
            const Scalar factor = Scalar(2) * smoothness * coefficientSquared;
            diagonal[pp] += factor;
            diagonal[pm] += factor;
            diagonal[mp] += factor;
            diagonal[mm] += factor;
        });
    return diagonal;
}

Scalar dot(const std::vector<Scalar>& a, const std::vector<Scalar>& b)
{
    // Последовательное накопление сохраняет округления и число PCG-итераций.
    long double result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<Scalar>(result);
}

Scalar euclideanNorm(const std::vector<Scalar>& values)
{
    return std::sqrt(std::max(Scalar(0), dot(values, values)));
}

// Матрично-свободный PCG: A=K+priorWeight*I+diag(Snap)+sum(rho*w*c*c^T).
// priorWeight>0 делает A положительно определенной; prior служит теплым стартом.
SolverResult solveLevel(
    const Grid& geometry,
    const std::vector<Scalar>& prior,
    const std::vector<Scalar>& extraDiagonal,
    const std::vector<Scalar>& extraRhs,
    const std::vector<PointConstraint>& finalConstraints,
    const ConvergentGriddingOptions& options)
{
    const std::size_t size = prior.size();
    const Scalar priorWeight = static_cast<Scalar>(options.priorWeight);
    const Scalar smoothness = static_cast<Scalar>(options.smoothness);
    const Scalar finalStrength = static_cast<Scalar>(options.finalPointStrength);

    std::vector<Scalar> rhs(size);
    const auto maskStorage = faultCurvatureMask(geometry);
    const auto* mask = geometry.faults ? &maskStorage : nullptr;
    std::vector<Scalar> diagonal = curvatureDiagonal(geometry, smoothness, mask);
    for (std::size_t i = 0; i < size; ++i) {
        diagonal[i] += priorWeight + extraDiagonal[i];
        rhs[i] = priorWeight * prior[i] + extraRhs[i];
        if (!finite(diagonal[i]) || !(diagonal[i] > 0) || !finite(rhs[i])) {
            throw std::runtime_error("level-system coefficients are not finite");
        }
    }
    for (const PointConstraint& constraint : finalConstraints) {
        const Scalar strength = finalStrength * constraint.weight;
        if (!finite(strength)) {
            throw std::runtime_error("point-constraint strength overflowed");
        }
        for (std::size_t a = 0; a < constraint.count; ++a) {
            const StencilTerm& term = constraint.terms[a];
            rhs[term.index] += strength * term.coefficient * constraint.value;
            diagonal[term.index] += strength * term.coefficient * term.coefficient;
            if (!finite(rhs[term.index]) || !finite(diagonal[term.index])) {
                throw std::runtime_error("point-constraint coefficients overflowed");
            }
        }
    }

    const auto applyA = [&](const std::vector<Scalar>& input,
                            std::vector<Scalar>& output) {
        std::fill(output.begin(), output.end(), Scalar(0));
        applyCurvature(geometry, input, output, smoothness, mask);
        for (std::size_t i = 0; i < size; ++i) {
            output[i] += (priorWeight + extraDiagonal[i]) * input[i];
        }
        for (const PointConstraint& constraint : finalConstraints) {
            const Scalar strength = finalStrength * constraint.weight;
            const Scalar projected = constraintValue(constraint, input);
            for (std::size_t i = 0; i < constraint.count; ++i) {
                const StencilTerm& term = constraint.terms[i];
                output[term.index] += strength * term.coefficient * projected;
            }
        }
    };

    std::vector<Scalar> x = prior;
    std::vector<Scalar> ax(size), residual(size), z(size), direction(size), ad(size);
    applyA(x, ax);
    for (std::size_t i = 0; i < size; ++i) {
        residual[i] = rhs[i] - ax[i];
        z[i] = residual[i] / diagonal[i];
    }
    direction = z;
    Scalar rz = dot(residual, z);
    const Scalar rhsNorm = euclideanNorm(rhs);
    const Scalar initialAxNorm = euclideanNorm(ax);
    const Scalar normalizer = std::max(rhsNorm, initialAxNorm);
    const Scalar tolerance = static_cast<Scalar>(options.relativeTolerance);
    const Scalar absoluteTolerance = static_cast<Scalar>(options.absoluteTolerance);
    // ||r|| <= absTol + relTol*max(||b||, ||A*x0||); масштаб фиксирован.
    const Scalar threshold = absoluteTolerance + tolerance * normalizer;
    if (!finite(normalizer) || !finite(threshold)) {
        throw std::runtime_error("PCG residual scale is not finite");
    }
    Scalar residualNorm = euclideanNorm(residual);
    Scalar relativeResidual = normalizer > 0 ? residualNorm / normalizer : Scalar(0);

    SolverResult result;
    result.relativeResidual = relativeResidual;
    result.converged = residualNorm <= threshold;
    if (result.converged) {
        result.values = std::move(x);
        return result;
    }

    for (std::size_t iteration = 0; iteration < options.maxSolverIterations; ++iteration) {
        applyA(direction, ad);
        const Scalar denominator = dot(direction, ad);
        if (!(denominator > 0) || !finite(denominator) || !finite(rz)) {
            throw std::runtime_error("PCG breakdown while solving the biharmonic system");
        }
        const Scalar alpha = rz / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error("PCG produced a non-finite step");
        }
        for (std::size_t i = 0; i < size; ++i) {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * ad[i];
        }

        // Каждые 50 шагов восстанавливаем истинную невязку и направление.
        const bool restartDirection = (iteration + 1) % 50 == 0;
        if (restartDirection) {
            applyA(x, ax);
            for (std::size_t i = 0; i < size; ++i) residual[i] = rhs[i] - ax[i];
        }
        residualNorm = euclideanNorm(residual);
        relativeResidual = normalizer > 0 ? residualNorm / normalizer : Scalar(0);
        result.iterations = iteration + 1;
        if (residualNorm <= threshold) {
            result.values = std::move(x);
            result.relativeResidual = relativeResidual;
            result.converged = true;
            return result;
        }

        for (std::size_t i = 0; i < size; ++i) z[i] = residual[i] / diagonal[i];
        const Scalar nextRz = dot(residual, z);
        if (!finite(nextRz)) {
            throw std::runtime_error("PCG produced a non-finite residual");
        }
        if (restartDirection) {
            direction = z;
        } else {
            const Scalar beta = nextRz / rz;
            for (std::size_t i = 0; i < size; ++i) {
                direction[i] = z[i] + beta * direction[i];
            }
        }
        rz = nextRz;
    }

    result.values = std::move(x);
    result.relativeResidual = relativeResidual;
    result.converged = false;
    return result;
}

Scalar maximumAbsolute(const std::vector<Scalar>& values)
{
    Scalar result = 0;
    for (Scalar value : values) {
        if (!finite(value)) {
            throw std::runtime_error("point-projection residual is not finite");
        }
        result = std::max(result, std::abs(value));
    }
    return result;
}

Scalar effectiveControlTolerance(
    const std::vector<PointConstraint>& constraints,
    const ConvergentGriddingOptions& options)
{
    Scalar valueScale = 1;
    for (const PointConstraint& constraint : constraints) {
        valueScale = std::max(valueScale, std::abs(constraint.value));
    }
    // Проверяем точность после приведения к публичному qreal.
    const Scalar roundingFloor = Scalar(64)
        * static_cast<Scalar>(std::numeric_limits<qreal>::epsilon()) * valueScale;
    return std::max(static_cast<Scalar>(options.controlTolerance), roundingFloor);
}

// G=C*C^T без полной матрицы: сжатая нумерация только узлов носителя C.
// Перенумерация сохраняет порядок строк, коэффициентов и арифметических операций.
class ConstraintGramOperator {
public:
    explicit ConstraintGramOperator(const std::vector<PointConstraint>& constraints)
        : rows_(constraints)
    {
        for (const PointConstraint& row : rows_) {
            for (std::size_t t = 0; t < row.count; ++t) {
                globalIndices_.push_back(row.terms[t].index);
            }
        }
        std::sort(globalIndices_.begin(), globalIndices_.end());
        globalIndices_.erase(std::unique(globalIndices_.begin(), globalIndices_.end()),
                             globalIndices_.end());
        for (PointConstraint& row : rows_) {
            for (std::size_t t = 0; t < row.count; ++t) {
                row.terms[t].index = static_cast<std::size_t>(std::lower_bound(
                    globalIndices_.begin(), globalIndices_.end(), row.terms[t].index)
                    - globalIndices_.begin());
            }
        }
        nodeWork_.resize(globalIndices_.size());
    }

    void apply(const std::vector<Scalar>& input, std::vector<Scalar>& output)
    {
        scatter(input);
        for (std::size_t i = 0; i < rows_.size(); ++i) {
            output[i] = constraintValue(rows_[i], nodeWork_);
            if (!finite(output[i])) {
                throw std::runtime_error("point-constraint Gram product overflowed");
            }
        }
    }

    void addCorrection(const std::vector<Scalar>& multiplier, std::vector<Scalar>& values)
    {
        scatter(multiplier);
        for (std::size_t i = 0; i < globalIndices_.size(); ++i) {
            values[globalIndices_[i]] += nodeWork_[i];
        }
        for (Scalar value : values) {
            if (!finite(value)) {
                throw std::runtime_error("exact-control correction produced a non-finite grid value");
            }
        }
    }

private:
    void scatter(const std::vector<Scalar>& input)
    {
        // Строки имеют общие узлы; scatter должен оставаться последовательным.
        std::fill(nodeWork_.begin(), nodeWork_.end(), Scalar(0));
        for (std::size_t i = 0; i < rows_.size(); ++i) {
            for (std::size_t t = 0; t < rows_[i].count; ++t) {
                const StencilTerm& term = rows_[i].terms[t];
                nodeWork_[term.index] += term.coefficient * input[i];
            }
        }
    }

    std::vector<PointConstraint> rows_;
    std::vector<std::size_t> globalIndices_;
    std::vector<Scalar> nodeWork_;
};

// Евклидова проекция: G*lambda=d-C*u, delta=C^T*lambda. Меняет только узлы C.
// Это отдельный Gram-PCG с абсолютным максимумом невязки; веса здесь не участвуют.
ProjectionResult projectOntoPointConstraints(
    Grid& grid,
    const std::vector<PointConstraint>& constraints,
    const ConvergentGriddingOptions& options)
{
    ProjectionResult result;
    result.converged = true;
    if (constraints.empty()) return result;

    const std::size_t count = constraints.size();
    const Scalar tolerance = effectiveControlTolerance(constraints, options);
    std::vector<Scalar> rhs(count), residual(count), inverseDiagonal(count);
    for (std::size_t i = 0; i < count; ++i) {
        rhs[i] = constraints[i].value
            - constraintValue(constraints[i], grid.values);
        Scalar diagonal = 0;
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const Scalar coefficient = constraints[i].terms[termIndex].coefficient;
            diagonal += coefficient * coefficient;
        }
        if (!finite(rhs[i]) || !finite(diagonal) || !(diagonal > 0)) {
            throw std::runtime_error("invalid exact point constraint");
        }
        inverseDiagonal[i] = Scalar(1) / diagonal;
    }

    result.maxError = maximumAbsolute(rhs);
    if (result.maxError <= tolerance) return result;

    std::vector<Scalar> multiplier(count, Scalar(0));
    residual = rhs;
    std::vector<Scalar> preconditioned(count), direction(count), gramDirection(count);
    std::vector<Scalar> gramMultiplier(count);
    ConstraintGramOperator gram(constraints);
    for (std::size_t i = 0; i < count; ++i) {
        preconditioned[i] = residual[i] * inverseDiagonal[i];
    }
    direction = preconditioned;
    Scalar residualPreconditioned = dot(residual, preconditioned);

    result.converged = false;
    for (std::size_t iteration = 0;
         iteration < options.maxControlProjectionIterations; ++iteration) {
        gram.apply(direction, gramDirection);
        const Scalar denominator = dot(direction, gramDirection);
        if (!(denominator > 0) || !finite(denominator)
            || !(residualPreconditioned > 0)
            || !finite(residualPreconditioned)) {
            throw std::runtime_error(
                "exact control equations are dependent or incompatible at this grid resolution");
        }
        const Scalar alpha = residualPreconditioned / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error("exact-control projection produced a non-finite step");
        }
        for (std::size_t i = 0; i < count; ++i) {
            multiplier[i] += alpha * direction[i];
            residual[i] -= alpha * gramDirection[i];
        }

        // Истинная невязка проверяется каждые 25 шагов и перед успешным выходом.
        const bool restartDirection = (iteration + 1) % 25 == 0;
        if (restartDirection || maximumAbsolute(residual) <= tolerance) {
            gram.apply(multiplier, gramMultiplier);
            for (std::size_t i = 0; i < count; ++i) {
                residual[i] = rhs[i] - gramMultiplier[i];
            }
        }

        result.iterations = iteration + 1;
        result.maxError = maximumAbsolute(residual);
        if (result.maxError <= tolerance) {
            result.converged = true;
            break;
        }

        for (std::size_t i = 0; i < count; ++i) {
            preconditioned[i] = residual[i] * inverseDiagonal[i];
        }
        const Scalar nextResidualPreconditioned = dot(residual, preconditioned);
        if (!(nextResidualPreconditioned > 0)
            || !finite(nextResidualPreconditioned)) {
            throw std::runtime_error(
                "exact control equations are dependent or incompatible at this grid resolution");
        }
        if (restartDirection) {
            direction = preconditioned;
        } else {
            const Scalar beta = nextResidualPreconditioned / residualPreconditioned;
            for (std::size_t i = 0; i < count; ++i) {
                direction[i] = preconditioned[i] + beta * direction[i];
            }
        }
        residualPreconditioned = nextResidualPreconditioned;
    }

    if (!result.converged) {
        throw std::runtime_error(
            "exact control projection did not converge; controls may be incompatible "
            "at this grid resolution");
    }

    gram.addCorrection(multiplier, grid.values);

    // Независимая проверка ограничений после изменения узлов сетки.
    result.maxError = 0;
    for (const PointConstraint& constraint : constraints) {
        result.maxError = std::max(result.maxError,
            std::abs(constraintValue(constraint, grid.values) - constraint.value));
    }
    if (result.maxError > tolerance) {
        throw std::runtime_error(
            "exact-control correction lost accuracy while updating the grid");
    }
    return result;
}

// Проверка входа и построение многоуровневой поверхности.

void validateOptions(const ConvergentGriddingOptions& options)
{
    const auto positiveFinite = [](qreal value) {
        return finite(static_cast<Scalar>(value)) && value > qreal(0);
    };
    const auto nonNegativeFinite = [](qreal value) {
        return finite(static_cast<Scalar>(value)) && value >= qreal(0);
    };

    if (options.initialSnapNodes == 0 || options.coarsestIntervals == 0
        || options.maxLevels == 0 || options.maxSolverIterations == 0) {
        throw std::invalid_argument("gridding counts must be greater than zero");
    }
    if (options.enforceExactControls
        && options.maxControlProjectionIterations == 0) {
        throw std::invalid_argument(
            "maxControlProjectionIterations must be positive in exact-control mode");
    }
    if (!nonNegativeFinite(options.smoothness)
        || !positiveFinite(options.priorWeight)
        || !nonNegativeFinite(options.snapStrength)
        || !nonNegativeFinite(options.finalPointStrength)
        || !positiveFinite(options.gaussianSigma)
        || !positiveFinite(options.relativeTolerance)
        || !nonNegativeFinite(options.absoluteTolerance)
        || !nonNegativeFinite(options.controlTolerance)) {
        throw std::invalid_argument("invalid convergent gridding numeric option");
    }
    const Scalar sigma = static_cast<Scalar>(options.gaussianSigma);
    if (!finite(sigma * sigma) || !(sigma * sigma > 0)) {
        throw std::invalid_argument("gaussianSigma squared must be finite and positive");
    }
    if (options.taylorOrder < 0 || options.taylorOrder > 2) {
        throw std::invalid_argument("taylorOrder must be 0, 1, or 2");
    }
}

std::vector<CanonicalPoint> validateInput(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options)
{
    // Сначала проверяются параметры, чтобы даже ветвь «нет активных точек»
    // имела единый контракт ошибок с обычным расчетом.
    validateOptions(options);

    if (surface.nx < 2 || surface.ny < 2) {
        throw std::invalid_argument("surface.nx and surface.ny must both be at least 2");
    }

    const Scalar minx = static_cast<Scalar>(surface.minx);
    const Scalar maxx = static_cast<Scalar>(surface.maxx);
    const Scalar miny = static_cast<Scalar>(surface.miny);
    const Scalar maxy = static_cast<Scalar>(surface.maxy);
    if (!finite(minx) || !finite(maxx) || !finite(miny) || !finite(maxy)
        || !(minx < maxx) || !(miny < maxy)) {
        throw std::invalid_argument("surface bounds must be finite and strictly increasing");
    }

    const Scalar spanX = maxx - minx;
    const Scalar spanY = maxy - miny;
    const Scalar dx = spanX / static_cast<Scalar>(surface.nx - 1);
    const Scalar dy = spanY / static_cast<Scalar>(surface.ny - 1);
    if (!finite(spanX) || !finite(spanY) || !finite(dx) || !finite(dy)
        || !(spanX > 0) || !(spanY > 0) || !(dx > 0) || !(dy > 0)
        || !finite(dx / dy) || !finite(dy / dx)) {
        throw std::invalid_argument("surface bounds produce an unsafe grid spacing");
    }

    const std::size_t expected = checkedNodeCount(surface.nx, surface.ny);
    if (surface.grid.size() != expected) {
        throw std::invalid_argument("surface.grid.size() must equal surface.nx * surface.ny");
    }
    for (qreal value : surface.grid) {
        if (!finite(static_cast<Scalar>(value))) {
            throw std::invalid_argument("surface.grid must contain only finite values");
        }
    }

    std::vector<CanonicalPoint> result;
    result.reserve(points.size());
    for (const Point& point : points) {
        const CanonicalPoint p{
            static_cast<Scalar>(point.x),
            static_cast<Scalar>(point.y),
            static_cast<Scalar>(point.value),
            static_cast<Scalar>(point.weight)};

        if (!finite(p.x) || !finite(p.y) || !finite(p.value) || !finite(p.weight)) {
            throw std::invalid_argument("control point fields must be finite");
        }
        if (p.weight < 0) {
            throw std::invalid_argument("control point weight must not be negative");
        }
        if (p.x < minx || p.x > maxx || p.y < miny || p.y > maxy) {
            throw std::invalid_argument("control point lies outside the surface bounds");
        }
        if (p.weight == 0) {
            continue;
        }
        result.push_back(p);
    }

    // Канонический порядок делает накопление воспроизводимым и не зависящим от
    // перестановки одного и того же набора входных точек.
    std::sort(result.begin(), result.end(), [](const CanonicalPoint& a, const CanonicalPoint& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        if (a.value != b.value) return a.value < b.value;
        return a.weight < b.weight;
    });

    std::vector<CanonicalPoint> grouped;
    grouped.reserve(result.size());
    for (std::size_t begin = 0; begin < result.size();) {
        std::size_t end = begin + 1;
        while (end < result.size()
               && result[end].x == result[begin].x
               && result[end].y == result[begin].y) {
            ++end;
        }
        long double weightSum = 0;
        long double weightedValue = 0;
        for (std::size_t i = begin; i < end; ++i) {
            weightSum += static_cast<long double>(result[i].weight);
            weightedValue += static_cast<long double>(result[i].weight)
                * static_cast<long double>(result[i].value);
        }
        const long double combinedValue = weightedValue / weightSum;
        if (!std::isfinite(weightSum) || !std::isfinite(combinedValue)
            || weightSum > static_cast<long double>(std::numeric_limits<Scalar>::max())
            || std::abs(combinedValue)
                > static_cast<long double>(std::numeric_limits<Scalar>::max())) {
            throw std::invalid_argument("co-located control point weights overflow");
        }
        grouped.push_back({result[begin].x, result[begin].y,
                           static_cast<Scalar>(combinedValue),
                           static_cast<Scalar>(weightSum)});
        begin = end;
    }
    result = std::move(grouped);

    Scalar maximumWeight = 0;
    for (const CanonicalPoint& point : result) {
        maximumWeight = std::max(maximumWeight, point.weight);
    }
    if (options.normalizePointWeights && maximumWeight > 0) {
        for (CanonicalPoint& point : result) {
            point.weight /= maximumWeight;
        }
    }
    return result;
}

// Создает пустой уровень с границами публичной Surface, но с заданным числом
// узлов. Все уровни поэтому покрывают одну физическую область.
Grid makeGridGeometry(const Surface& surface, std::size_t nx, std::size_t ny)
{
    Grid result;
    result.nx = nx;
    result.ny = ny;
    result.minx = static_cast<Scalar>(surface.minx);
    result.maxx = static_cast<Scalar>(surface.maxx);
    result.miny = static_cast<Scalar>(surface.miny);
    result.maxy = static_cast<Scalar>(surface.maxy);
    result.values.resize(checkedNodeCount(nx, ny));
    return result;
}

// Координаты одной билинейной ячейки, включая замкнутую верхнюю границу.
struct Cell {
    std::size_t ix0, iy0, ix1, iy1;
    Scalar tx, ty;
};

Cell cellAt(const Grid& grid, Scalar x, Scalar y)
{
    const Scalar fx = std::clamp((x - grid.minx) / grid.dx(), Scalar(0),
                                 static_cast<Scalar>(grid.nx - 1));
    const Scalar fy = std::clamp((y - grid.miny) / grid.dy(), Scalar(0),
                                 static_cast<Scalar>(grid.ny - 1));
    const auto ix = static_cast<std::size_t>(std::floor(fx));
    const auto iy = static_cast<std::size_t>(std::floor(fy));
    return {ix, iy, std::min(ix + 1, grid.nx - 1), std::min(iy + 1, grid.ny - 1),
            fx - static_cast<Scalar>(ix), fy - static_cast<Scalar>(iy)};
}

// Нулевые веса пропускаются, совпавшие индексы углов объединяются.
void addTerm(PointConstraint& constraint, std::size_t index, Scalar coefficient)
{
    if (coefficient == 0) return;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        if (constraint.terms[i].index == index) {
            constraint.terms[i].coefficient += coefficient;
            return;
        }
    }
    constraint.terms[constraint.count++] = {index, coefficient};
}

// Выбираем видимые и взаимно совместимые углы с максимальной суммой весов.
bool samplingConstraint(const Grid& grid, Scalar x, Scalar y,
                        PointConstraint& result)
{
    result = {};
    if (grid.faults && grid.faults->onFault(x, y)) return false;
    const auto [ix, iy, ix1, iy1, tx, ty] = cellAt(grid, x, y);
    PointConstraint corners;
    addTerm(corners, grid.index(ix, iy), (1 - tx) * (1 - ty));
    addTerm(corners, grid.index(ix1, iy), tx * (1 - ty));
    addTerm(corners, grid.index(ix, iy1), (1 - tx) * ty);
    addTerm(corners, grid.index(ix1, iy1), tx * ty);
    if (!grid.faults) { result = corners; return true; }

    unsigned visible = 0;
    for (std::size_t k = 0; k < corners.count; ++k) {
        const auto index = corners.terms[k].index;
        if (grid.faults->visible(x, y, grid.x(index % grid.nx), grid.y(index / grid.nx)))
            visible |= 1u << k;
    }
    unsigned best = 0;
    Scalar bestWeight = 0;
    // Каждая пара углов встречается в нескольких из 15 подмножеств.
    // Проверяем геометрию пары только один раз, сохраняя прежний порядок
    // перебора подмножеств и сложения весов (результат выборки не меняется).
    std::array<unsigned, 4> compatibleWith{};
    for (std::size_t a = 0; a < corners.count; ++a) {
        if (!(visible & (1u << a))) continue;
        for (std::size_t b = a + 1; b < corners.count; ++b) {
            if ((visible & (1u << b)) && visibleNodes(grid, corners.terms[a].index,
                                                        corners.terms[b].index))
                compatibleWith[a] |= 1u << b;
        }
    }
    for (unsigned mask = 1; mask < (1u << corners.count); ++mask) {
        if ((mask & visible) != mask) continue;
        bool compatible = true;
        Scalar weight = 0;
        for (std::size_t a = 0; a < corners.count && compatible; ++a) {
            if (!(mask & (1u << a))) continue;
            weight += corners.terms[a].coefficient;
            for (std::size_t b = a + 1; b < corners.count; ++b) {
                if ((mask & (1u << b)) && !(compatibleWith[a] & (1u << b))) {
                    compatible = false;
                    break;
                }
            }
        }
        if (compatible && weight > bestWeight) { best = mask; bestWeight = weight; }
    }
    if (bestWeight > 0) {
        for (std::size_t k = 0; k < corners.count; ++k) {
            if (best & (1u << k)) addTerm(result, corners.terms[k].index,
                                         corners.terms[k].coefficient / bestWeight);
        }
        return true;
    }

    // Без видимых углов допускается только локальная опора в двух ячейках.
    const Scalar fx = std::clamp((x - grid.minx) / grid.dx(), Scalar(0),
                                 static_cast<Scalar>(grid.nx - 1));
    const Scalar fy = std::clamp((y - grid.miny) / grid.dy(), Scalar(0),
                                 static_cast<Scalar>(grid.ny - 1));
    Scalar bestDistance = std::numeric_limits<Scalar>::infinity();
    std::size_t bestIndex = 0;
    for (std::size_t j = iy > 2 ? iy - 2 : 0; j <= std::min(iy + 2, grid.ny - 1); ++j) {
        for (std::size_t i = ix > 2 ? ix - 2 : 0; i <= std::min(ix + 2, grid.nx - 1); ++i) {
            const Scalar dx = static_cast<Scalar>(i) - fx;
            const Scalar dy = static_cast<Scalar>(j) - fy;
            const Scalar distance = dx * dx + dy * dy;
            if (distance < bestDistance && grid.faults->visible(x, y, grid.x(i), grid.y(j))) {
                bestDistance = distance;
                bestIndex = grid.index(i, j);
            }
        }
    }
    if (!finite(bestDistance)) return false;
    addTerm(result, bestIndex, Scalar(1));
    return true;
}

Scalar rawBilinearSample(const Grid& grid, Scalar x, Scalar y)
{
    const auto [ix0, iy0, ix1, iy1, tx, ty] = cellAt(grid, x, y);
    const Scalar v00 = grid.values[grid.index(ix0, iy0)];
    const Scalar v10 = grid.values[grid.index(ix1, iy0)];
    const Scalar v01 = grid.values[grid.index(ix0, iy1)];
    const Scalar v11 = grid.values[grid.index(ix1, iy1)];
    return (Scalar(1) - tx) * (Scalar(1) - ty) * v00
         + tx * (Scalar(1) - ty) * v10
         + (Scalar(1) - tx) * ty * v01
         + tx * ty * v11;
}

Scalar bilinearSample(const Grid& grid, Scalar x, Scalar y)
{
    if (!grid.faults) return rawBilinearSample(grid, x, y);
    PointConstraint row;
    if (!samplingConstraint(grid, x, y, row)) {
        throw std::invalid_argument(
            "cannot sample on a fault or in a fault block without local grid nodes; refine nx/ny");
    }
    return constraintValue(row, grid.values);
}

Grid resample(const Grid& source, std::size_t nx, std::size_t ny)
{
    Grid result = gridGeometry(source);
    result.nx = nx;
    result.ny = ny;
    result.values.assign(checkedNodeCount(nx, ny), Scalar(0));

    for (std::size_t iy = 0; iy < ny; ++iy) {
        const Scalar ty = static_cast<Scalar>(iy) / static_cast<Scalar>(ny - 1);
        // С разломами один и тот же способ вычисления координат обязателен
        // для resample, геометрических масок и Snap: разные порядки округления
        // на больших XY могут переместить пограничный узел на другую сторону.
        const Scalar y = source.faults ? result.y(iy) : (iy + 1 == ny
            ? result.maxy
            : result.miny + ty * (result.maxy - result.miny));
        for (std::size_t ix = 0; ix < nx; ++ix) {
            const Scalar tx = static_cast<Scalar>(ix) / static_cast<Scalar>(nx - 1);
            const Scalar x = source.faults ? result.x(ix) : (ix + 1 == nx
                ? result.maxx
                : result.minx + tx * (result.maxx - result.minx));
            // Узел ровно на разломе не представляет ни одну из двух сторон.
            // Его буферное значение не участвует в связях/выборках; сохраняем
            // обычный prior, чтобы grid оставался конечным и без NaN-масок.
            result.values[result.index(ix, iy)] = source.faults && source.faults->onFault(x, y)
                ? rawBilinearSample(source, x, y) : bilinearSample(source, x, y);
        }
    }
    return result;
}

// Копирует публичный qreal-вектор во внутренний double, не изменяя row-major
// порядок и геометрию.
Grid surfaceAsGrid(const Surface& surface)
{
    Grid result = makeGridGeometry(surface, surface.nx, surface.ny);
    std::transform(surface.grid.begin(), surface.grid.end(), result.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    return result;
}

// Целочисленное деление вверх (ceiling division) без сложения value+divisor-1,
// которое могло бы переполнить size_t.
std::size_t ceilDivide(std::size_t value, std::size_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

std::vector<std::pair<std::size_t, std::size_t>> buildHierarchy(
    std::size_t finalNx,
    std::size_t finalNy,
    const ConvergentGriddingOptions& options)
{
    std::vector<std::size_t> factors{1};
    std::size_t factor = 1;
    while (factors.size() < options.maxLevels) {
        const std::size_t intervalsX = ceilDivide(finalNx - 1, factor);
        const std::size_t intervalsY = ceilDivide(finalNy - 1, factor);
        if (intervalsX <= options.coarsestIntervals
            && intervalsY <= options.coarsestIntervals) {
            break;
        }
        if (factor > std::numeric_limits<std::size_t>::max() / 2) {
            break;
        }
        factor *= 2;
        factors.push_back(factor);
    }

    std::vector<std::pair<std::size_t, std::size_t>> hierarchy;
    hierarchy.reserve(factors.size());
    for (auto it = factors.rbegin(); it != factors.rend(); ++it) {
        const std::size_t nx = ceilDivide(finalNx - 1, *it) + 1;
        const std::size_t ny = ceilDivide(finalNy - 1, *it) + 1;
        if (hierarchy.empty() || hierarchy.back() != std::make_pair(nx, ny)) {
            hierarchy.emplace_back(nx, ny);
        }
    }
    if (hierarchy.empty() || hierarchy.back() != std::make_pair(finalNx, finalNy)) {
        hierarchy.emplace_back(finalNx, finalNy);
    }
    return hierarchy;
}

// Проверка представимости (representability): каждый активный узел целевого
// уровня должен иметь одностороннюю опору на исходном. Проверяются именно
// соседние переходы, поскольку для 512 узлов уровни не строго вложены.
bool supportsTransfer(const Grid& source, const Grid& target)
{
    PointConstraint row;
    for (std::size_t j = 0; j < target.ny; ++j) {
        for (std::size_t i = 0; i < target.nx; ++i) {
            const Scalar x = target.x(i), y = target.y(j);
            if (source.faults->onFault(x, y)) continue;
            if (!samplingConstraint(source, x, y, row)) return false;
        }
    }
    return true;
}

std::vector<std::pair<std::size_t, std::size_t>> faultHierarchy(
    const Surface& surface, const Grid& input,
    const std::vector<CanonicalPoint>& controls,
    const ConvergentGriddingOptions& options)
{
    auto hierarchy = buildHierarchy(surface.nx, surface.ny, options);
    if (!input.faults) return hierarchy;
    std::size_t first = 0;
    for (std::size_t level = 0; level < hierarchy.size(); ++level) {
        Grid current = makeGridGeometry(surface, hierarchy[level].first, hierarchy[level].second);
        current.faults = input.faults;
        PointConstraint row;
        bool supported = true;
        for (const auto& point : controls) {
            if (!samplingConstraint(current, point.x, point.y, row)) { supported = false; break; }
        }
        if (supported && level + 1 < hierarchy.size()) {
            Grid next = makeGridGeometry(surface, hierarchy[level + 1].first,
                                         hierarchy[level + 1].second);
            next.faults = input.faults;
            supported = supportsTransfer(current, next);
        }
        if (supported) supported = supportsTransfer(input, current);
        if (!supported) first = level + 1;
    }
    if (first == hierarchy.size()) {
        throw std::invalid_argument(
            "a control point is in a fault block without local grid nodes; refine nx/ny");
    }
    hierarchy.erase(hierarchy.begin(), hierarchy.begin() + static_cast<std::ptrdiff_t>(first));
    return hierarchy;
}

Scalar valueAt(const Grid& grid, std::size_t ix, std::size_t iy)
{
    return grid.values[grid.index(ix, iy)];
}

Derivatives derivativesAtNode(const Grid& grid, std::size_t ix, std::size_t iy)
{
    const Scalar hx = grid.dx();
    const Scalar hy = grid.dy();
    Derivatives d;

    if (grid.faults) {
        const auto center = grid.index(ix, iy);
        if (grid.faults->onFault(grid.x(ix), grid.y(iy))) return d;
        const auto along = [&](bool alongX, Scalar& first, Scalar& second) {
            const auto position = alongX ? ix : iy;
            const auto size = alongX ? grid.nx : grid.ny;
            const Scalar step = alongX ? hx : hy;
            const auto indexAt = [&](std::size_t n) {
                return alongX ? grid.index(n, iy) : grid.index(ix, n);
            };
            const auto safe = [&](std::size_t n) { return visibleNodes(grid, center, indexAt(n)); };
            const auto v = [&](std::size_t n) { return grid.values[indexAt(n)]; };
            const Scalar u = grid.values[center];
            const bool left = position > 0 && safe(position - 1);
            const bool right = position + 1 < size && safe(position + 1);
            if (left && right) {
                first = (v(position + 1) - v(position - 1)) / (2 * step);
                second = (v(position - 1) - 2 * u + v(position + 1)) / (step * step);
            } else if (right) {
                first = (v(position + 1) - u) / step;
                if (position + 2 < size && safe(position + 2))
                    second = (u - 2 * v(position + 1) + v(position + 2)) / (step * step);
            } else if (left) {
                first = (u - v(position - 1)) / step;
                if (position >= 2 && safe(position - 2))
                    second = (u - 2 * v(position - 1) + v(position - 2)) / (step * step);
            }
        };
        along(true, d.gx, d.gxx);
        along(false, d.gy, d.gyy);
        const auto x0 = ix > 0 && visibleNodes(grid, center, grid.index(ix - 1, iy)) ? ix - 1 : ix;
        const auto x1 = ix + 1 < grid.nx && visibleNodes(grid, center, grid.index(ix + 1, iy)) ? ix + 1 : ix;
        const auto y0 = iy > 0 && visibleNodes(grid, center, grid.index(ix, iy - 1)) ? iy - 1 : iy;
        const auto y1 = iy + 1 < grid.ny && visibleNodes(grid, center, grid.index(ix, iy + 1)) ? iy + 1 : iy;
        if (x1 > x0 && y1 > y0) {
            const std::array<std::size_t, 4> corners{
                grid.index(x0, y0), grid.index(x1, y0), grid.index(x0, y1), grid.index(x1, y1)};
            bool safe = true;
            for (std::size_t a = 0; a < 4 && safe; ++a) {
                if (!visibleNodes(grid, center, corners[a])) safe = false;
                for (std::size_t b = a + 1; b < 4 && safe; ++b)
                    if (!visibleNodes(grid, corners[a], corners[b])) safe = false;
            }
            if (safe) d.gxy = (grid.values[corners[3]] - grid.values[corners[1]]
                               - grid.values[corners[2]] + grid.values[corners[0]])
                / (static_cast<Scalar>(x1 - x0) * hx * static_cast<Scalar>(y1 - y0) * hy);
        }
        return d;
    }

    const auto along = [&](bool alongX, Scalar& first, Scalar& second) {
        const auto p = alongX ? ix : iy;
        const auto size = alongX ? grid.nx : grid.ny;
        const Scalar step = alongX ? hx : hy;
        const auto v = [&](std::size_t n) {
            return alongX ? valueAt(grid, n, iy) : valueAt(grid, ix, n);
        };
        if (p == 0) first = (v(1) - v(0)) / step;
        else if (p + 1 == size) first = (v(p) - v(p - 1)) / step;
        else first = (v(p + 1) - v(p - 1)) / (Scalar(2) * step);
        if (size >= 3) {
            if (p == 0) second = (v(0) - Scalar(2) * v(1) + v(2)) / (step * step);
            else if (p + 1 == size)
                second = (v(p) - Scalar(2) * v(p - 1) + v(p - 2)) / (step * step);
            else second = (v(p - 1) - Scalar(2) * v(p) + v(p + 1)) / (step * step);
        }
    };
    along(true, d.gx, d.gxx);
    along(false, d.gy, d.gyy);

    const std::size_t ix0 = ix == 0 ? 0 : ix - 1;
    const std::size_t ix1 = ix + 1 < grid.nx ? ix + 1 : ix;
    const std::size_t iy0 = iy == 0 ? 0 : iy - 1;
    const std::size_t iy1 = iy + 1 < grid.ny ? iy + 1 : iy;
    const Scalar spanX = static_cast<Scalar>(ix1 - ix0) * hx;
    const Scalar spanY = static_cast<Scalar>(iy1 - iy0) * hy;
    if (spanX > 0 && spanY > 0) {
        d.gxy = (valueAt(grid, ix1, iy1) - valueAt(grid, ix1, iy0)
                 - valueAt(grid, ix0, iy1) + valueAt(grid, ix0, iy0))
            / (spanX * spanY);
    }
    return d;
}

Derivatives derivativesAt(const Grid& grid, Scalar x, Scalar y)
{
    if (grid.faults) {
        PointConstraint row;
        if (!samplingConstraint(grid, x, y, row))
            throw std::invalid_argument("Taylor derivatives have no visible grid support; refine nx/ny");
        Derivatives result;
        for (std::size_t k = 0; k < row.count; ++k) {
            const auto& term = row.terms[k];
            const auto d = derivativesAtNode(grid, term.index % grid.nx, term.index / grid.nx);
            result.gx += term.coefficient * d.gx;
            result.gy += term.coefficient * d.gy;
            result.gxx += term.coefficient * d.gxx;
            result.gxy += term.coefficient * d.gxy;
            result.gyy += term.coefficient * d.gyy;
        }
        return result;
    }
    const auto [ix0, iy0, ix1, iy1, tx, ty] = cellAt(grid, x, y);

    const Derivatives d00 = derivativesAtNode(grid, ix0, iy0);
    const Derivatives d10 = derivativesAtNode(grid, ix1, iy0);
    const Derivatives d01 = derivativesAtNode(grid, ix0, iy1);
    const Derivatives d11 = derivativesAtNode(grid, ix1, iy1);
    const auto blend = [&](Scalar Derivatives::*member) {
        return (Scalar(1) - tx) * (Scalar(1) - ty) * d00.*member
             + tx * (Scalar(1) - ty) * d10.*member
             + (Scalar(1) - tx) * ty * d01.*member
             + tx * ty * d11.*member;
    };
    return {blend(&Derivatives::gx), blend(&Derivatives::gy),
            blend(&Derivatives::gxx), blend(&Derivatives::gxy),
            blend(&Derivatives::gyy)};
}

struct CandidateNode {
    std::size_t index{};
    std::size_t ix{};
    std::size_t iy{};
    Scalar distanceSquared{};
};

std::vector<CandidateNode> nearestNodes(
    const Grid& grid, Scalar x, Scalar y, std::size_t requested)
{
    requested = std::min(requested, grid.values.size());
    const Scalar fx = (x - grid.minx) / grid.dx();
    const Scalar fy = (y - grid.miny) / grid.dy();
    const std::size_t centerX = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fx, Scalar(0), static_cast<Scalar>(grid.nx - 1)))), grid.nx - 1);
    const std::size_t centerY = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fy, Scalar(0), static_cast<Scalar>(grid.ny - 1)))), grid.ny - 1);
    const std::size_t radius = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<Scalar>(requested)))) + 3;
    const std::size_t minX = centerX > radius ? centerX - radius : 0;
    const std::size_t minY = centerY > radius ? centerY - radius : 0;
    const std::size_t maxX = std::min(centerX + radius, grid.nx - 1);
    const std::size_t maxY = std::min(centerY + radius, grid.ny - 1);

    std::vector<CandidateNode> candidates;
    const auto collect = [&](std::size_t x0, std::size_t y0,
                             std::size_t x1, std::size_t y1) {
        candidates.reserve((x1 - x0 + 1) * (y1 - y0 + 1));
        for (std::size_t iy = y0; iy <= y1; ++iy) {
            for (std::size_t ix = x0; ix <= x1; ++ix) {
                if (grid.faults && !grid.faults->visible(x, y, grid.x(ix), grid.y(iy))) continue;
                const Scalar dxCells = static_cast<Scalar>(ix) - fx;
                const Scalar dyCells = static_cast<Scalar>(iy) - fy;
                candidates.push_back({grid.index(ix, iy), ix, iy,
                                      dxCells * dxCells + dyCells * dyCells});
            }
        }
    };
    collect(minX, minY, maxX, maxY);

    // Для очень тонкой сетки или точки у края локального квадрата может быть
    // недостаточно. Редкий полный просмотр сохраняет точный результат именно
    // поиска ближайших узлов (это не относится к exact-интерполяции значений).
    if (candidates.size() < requested && !grid.faults) {
        candidates.clear();
        collect(0, 0, grid.nx - 1, grid.ny - 1);
    }

    // Нужны только первые requested кандидатов. Частичная сортировка
    // (partial sort) сохраняет тот же упорядоченный префикс, но не сортирует
    // отброшенный хвост. Пара (distanceSquared, index) задает полный порядок.
    const std::size_t selected = std::min(requested, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + selected,
                     candidates.end(), [](const CandidateNode& a,
                                                       const CandidateNode& b) {
        if (a.distanceSquared != b.distanceSquared) {
            return a.distanceSquared < b.distanceSquared;
        }
        return a.index < b.index;
    });
    if (candidates.empty()) {
        throw std::invalid_argument("control point has no visible Snap nodes; refine nx/ny");
    }
    // Барьер может оставить меньше узлов, чем номинальное расписание Snap.
    // Используем доступные, не расширяя связь через разлом ради нужного числа.
    candidates.resize(selected);
    return candidates;
}

std::size_t snapNodeCount(
    const Grid& grid,
    const Grid& coarsest,
    bool finalLevel,
    const ConvergentGriddingOptions& options)
{
    if (finalLevel) return 1;
    const std::size_t cappedInitial = std::min(
        options.initialSnapNodes, grid.values.size());
    const long double scaleSquared =
        static_cast<long double>(grid.dx() / coarsest.dx())
        * static_cast<long double>(grid.dy() / coarsest.dy());
    const long double scaled = static_cast<long double>(cappedInitial)
        * std::sqrt(scaleSquared);
    if (!std::isfinite(scaled) || scaled < 0
        || scaled > static_cast<long double>(grid.values.size())) {
        throw std::invalid_argument("initialSnapNodes cannot be scaled safely");
    }
    return std::max<std::size_t>(1,
        static_cast<std::size_t>(std::floor(scaled + 0.5L)));
}

void addSnapConstraints(
    const Grid& prior,
    const std::vector<CanonicalPoint>& points,
    std::size_t count,
    const ConvergentGriddingOptions& options,
    std::vector<Scalar>& diagonal,
    std::vector<Scalar>& rhs)
{
    std::vector<Scalar> accumulatedWeight(prior.values.size(), Scalar(0));
    std::vector<Scalar> accumulatedValue(prior.values.size(), Scalar(0));
    const Scalar sigma2 = static_cast<Scalar>(options.gaussianSigma)
        * static_cast<Scalar>(options.gaussianSigma);

    for (const CanonicalPoint& point : points) {
        const Derivatives d = derivativesAt(prior, point.x, point.y);
        const std::vector<CandidateNode> nodes = nearestNodes(prior, point.x, point.y, count);
        std::vector<Scalar> kernels(nodes.size());
        Scalar kernelSum = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            // Вычитание ближайшего r^2 сокращается при нормировке и защищает
            // fault-вариант от underflow, когда ближайший видимый узел дальше.
            const Scalar shifted = nodes[i].distanceSquared
                - (prior.faults ? nodes.front().distanceSquared : Scalar(0));
            kernels[i] = std::exp(-shifted / (Scalar(2) * sigma2));
            kernelSum += kernels[i];
        }
        if (!(kernelSum > 0) || !finite(kernelSum)) {
            throw std::runtime_error("failed to normalize the Snap distance kernel");
        }

        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const CandidateNode& node = nodes[i];
            const Scalar nodeX = prior.faults ? prior.x(node.ix)
                : prior.minx + static_cast<Scalar>(node.ix) * prior.dx();
            const Scalar nodeY = prior.faults ? prior.y(node.iy)
                : prior.miny + static_cast<Scalar>(node.iy) * prior.dy();
            const Scalar deltaX = nodeX - point.x;
            const Scalar deltaY = nodeY - point.y;
            Scalar projected = point.value;
            if (options.taylorOrder >= 1) {
                projected += d.gx * deltaX + d.gy * deltaY;
            }
            if (options.taylorOrder >= 2) {
                projected += Scalar(0.5) * (d.gxx * deltaX * deltaX
                    + Scalar(2) * d.gxy * deltaX * deltaY
                    + d.gyy * deltaY * deltaY);
            }
            if (!finite(projected)) {
                throw std::runtime_error("Taylor projection produced a non-finite value");
            }

            const Scalar weight = point.weight * kernels[i] / kernelSum;
            const Scalar nextWeight = accumulatedWeight[node.index] + weight;
            const Scalar nextValue = accumulatedValue[node.index] + weight * projected;
            if (!finite(weight) || !finite(nextWeight) || !finite(nextValue)) {
                throw std::runtime_error("Snap constraint accumulation overflowed");
            }
            accumulatedWeight[node.index] = nextWeight;
            accumulatedValue[node.index] = nextValue;
        }
    }

    const Scalar strength = static_cast<Scalar>(options.snapStrength);
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
        diagonal[i] += strength * accumulatedWeight[i];
        rhs[i] += strength * accumulatedValue[i];
        if (!finite(diagonal[i]) || !finite(rhs[i])) {
            throw std::runtime_error("scaled Snap constraint overflowed");
        }
    }
}

// Единый оператор C для мягких и точных ограничений и публичной выборки.
std::vector<PointConstraint> makePointConstraints(
    const Grid& grid, const std::vector<CanonicalPoint>& points)
{
    std::vector<PointConstraint> result;
    result.reserve(points.size());
    for (const CanonicalPoint& point : points) {
        PointConstraint constraint;
        if (!samplingConstraint(grid, point.x, point.y, constraint))
            throw std::invalid_argument("control point has no fault-aware interpolation support; refine nx/ny");
        constraint.value = point.value;
        constraint.weight = point.weight;
        result.push_back(constraint);
    }
    return result;
}

// Диагностическая infinity-норма |C*u-d| по каноническим точкам. При наличии
// разломов C совпадает с fault-aware sampleSurface. Она считается
// одинаково в soft- и exact-режиме и не является residual основного solver.
Scalar maxControlError(const Grid& grid, const std::vector<CanonicalPoint>& points)
{
    Scalar result = 0;
    for (const CanonicalPoint& point : points) {
        result = std::max(result, std::abs(bilinearSample(grid, point.x, point.y) - point.value));
    }
    return result;
}

} // namespace

// Оркестратор полной схемы, приведенной в начале файла. Все промежуточные
// объекты локальны, поэтому исключение до последнего присваивания не меняет
// переданную пользователем Surface.
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options)
{
    return convergentGridding(surface, points, std::vector<Fault>{}, options);
}

ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options)
{
    const std::vector<CanonicalPoint> controls = validateInput(surface, points, options);
    const FaultGeometry barriers(surface, faults);
    for (const auto& point : controls) {
        if (barriers.onFault(point.x, point.y))
            throw std::invalid_argument("a control point lies on a fault; specify a point on one side");
    }
    ConvergentGriddingReport report;
    if (controls.empty()) {
        // Валидная поверхность без ненулевых контрольных точек не изменяется.
        return report;
    }

    // Самый грубый prior получается ресемплированием входной surface.grid.
    // Отдельный тренд по точкам здесь не строится. Иерархия содержит общие
    // min/max и заканчивается строго исходными surface.nx/surface.ny.
    Grid input = surfaceAsGrid(surface);
    input.faults = barriers.empty() ? nullptr : &barriers;
    const auto hierarchy = faultHierarchy(surface, input, controls, options);
    Grid coarsest = input.faults && hierarchy.front() == std::make_pair(surface.nx, surface.ny)
        ? input : resample(input, hierarchy.front().first, hierarchy.front().second);
    Grid solved = std::move(coarsest);
    // Для правила Snap нужны только шаги самого грубого уровня.
    coarsest = gridGeometry(solved);

    for (std::size_t levelIndex = 0; levelIndex < hierarchy.size(); ++levelIndex) {
        const auto [nx, ny] = hierarchy[levelIndex];
        // Уточнение (Refine/prolongation): первый prior уже имеет нужный грубый
        // размер; далее решение предыдущего уровня билинейно переносится на
        // более частую сетку.
        Grid prior = levelIndex == 0 ? std::move(solved) : resample(solved, nx, ny);
        const bool finalLevel = levelIndex + 1 == hierarchy.size();
        const std::size_t snapNodes = snapNodeCount(prior, coarsest, finalLevel, options);

        // Привязка Snap: контрольные значения проецируются по Тейлору (Taylor
        // projection) в ближайшие узлы и превращаются в диагональные мягкие
        // штрафы текущего уровня.
        std::vector<Scalar> snapDiagonal(prior.values.size(), Scalar(0));
        std::vector<Scalar> snapRhs(prior.values.size(), Scalar(0));
        addSnapConstraints(prior, controls, snapNodes, options, snapDiagonal, snapRhs);

        // На финальном уровне добавляется мягкая билинейная привязка C_p*u=d_p.
        // Она уменьшает величину последующей exact-поправки, но не заменяет ее.
        const std::vector<PointConstraint> finalConstraints = finalLevel
            ? makePointConstraints(prior, controls)
            : std::vector<PointConstraint>{};

        SolverResult level = solveLevel(prior, prior.values, snapDiagonal, snapRhs,
                                        finalConstraints, options);
        report.levels.push_back({nx, ny, snapNodes, level.iterations,
                                 static_cast<qreal>(level.relativeResidual), level.converged});
        if (!level.converged && options.throwOnNonConvergence) {
            // Термин biharmonic указывает на доминирующий оператор K, хотя A
            // также содержит prior и точечные penalties. Размер в сообщении —
            // текущий промежуточный level, например 33x33, а не итоговый grid.
            throw std::runtime_error("biharmonic PCG did not converge at grid level "
                + std::to_string(nx) + "x" + std::to_string(ny));
        }
        solved = std::move(prior);
        solved.values = std::move(level.values);
    }

    // Мягкое решение (soft solution) обычно лишь приближенно выполняет C*u=d.
    // Необязательная exact-control projection доводит билинейные значения до
    // effective tolerance отдельным решением в пространстве точек.
    const std::vector<PointConstraint> outputConstraints =
        makePointConstraints(solved, controls);
    if (options.enforceExactControls) {
        const ProjectionResult projection = projectOntoPointConstraints(
            solved, outputConstraints, options);
        report.controlProjectionIterations = projection.iterations;
    }

    if (input.faults) {
        // Изолированные узлы на геометрическом разломе не определяют ни один
        // берег. Для стабильного хранения оставляем именно исходное значение;
        // fault-aware sampler никогда не использует его как опорное.
        for (std::size_t j = 0; j < solved.ny; ++j)
            for (std::size_t i = 0; i < solved.nx; ++i)
                if (input.faults->onFault(solved.x(i), solved.y(j)))
                    solved.values[solved.index(i, j)] = input.values[input.index(i, j)];
    }

    // Публичный qreal может иметь меньшую разрядность, чем внутренний double,
    // поэтому преобразование выполняется во временный буфер и проверяется до
    // изменения surface.grid.
    std::vector<qreal> output(solved.values.size());
    std::transform(solved.values.begin(), solved.values.end(), output.begin(),
        [](Scalar value) {
            if (!finite(value)) {
                throw std::runtime_error("convergent gridding produced a non-finite result");
            }
            const qreal converted = static_cast<qreal>(value);
            if (!finite(static_cast<Scalar>(converted))) {
                throw std::runtime_error("convergent gridding result does not fit in qreal");
            }
            return converted;
        });

    // Отчет измеряет невязку именно возвращаемых значений, включая возможную
    // потерю точности, когда qreal в Qt-сборке имеет меньшую разрядность, чем
    // внутренний Scalar.
    std::transform(output.begin(), output.end(), solved.values.begin(),
        [](qreal value) { return static_cast<Scalar>(value); });
    report.maxControlError = static_cast<qreal>(maxControlError(solved, controls));
    const Scalar acceptedControlError = effectiveControlTolerance(
        outputConstraints, options);
    report.controlsSatisfied = static_cast<Scalar>(report.maxControlError)
        <= acceptedControlError;
    // В soft-режиме false — допустимый диагностический результат. В exact-
    // режиме это означает, что cast в qreal разрушил уже выполненную коррекцию.
    if (options.enforceExactControls && !report.controlsSatisfied) {
        throw std::runtime_error(
            "qreal precision is insufficient to retain the exact control-point correction");
    }

    // Единственная запись в объект пользователя: все вычисления, exact-проекция,
    // преобразование и проверки завершены. Это строгая гарантия исключений
    // (strong exception guarantee), но не atomic/thread-safe операция.
    surface.grid = std::move(output);
    return report;
}

Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options,
    ConvergentGriddingReport* report)
{
    // Немутирующая перегрузка реализована через локальную копию и основной
    // оркестратор, поэтому численный путь обеих публичных функций идентичен.
    return convergentGriddedSurface(surface, points, {}, options, report);
}

Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options,
    ConvergentGriddingReport* report)
{
    Surface result = surface;
    auto localReport = convergentGridding(result, points, faults, options);
    if (report) *report = std::move(localReport);
    return result;
}

qreal sampleSurface(const Surface& surface, qreal x, qreal y,
                    const std::vector<Fault>& faults)
{
    // Для одной выборки не копируем и не сканируем весь grid: после проверки
    // геометрии читаются только значения фактического support (до 4 узлов).
    if (surface.nx < 2 || surface.ny < 2
        || surface.grid.size() != checkedNodeCount(surface.nx, surface.ny))
        throw std::invalid_argument("invalid surface dimensions for sampling");
    Grid grid;
    grid.nx = surface.nx; grid.ny = surface.ny;
    grid.minx = surface.minx; grid.maxx = surface.maxx;
    grid.miny = surface.miny; grid.maxy = surface.maxy;
    if (!finite(grid.minx) || !finite(grid.maxx) || !finite(grid.miny) || !finite(grid.maxy)
        || !(grid.maxx > grid.minx) || !(grid.maxy > grid.miny)
        || !finite(grid.dx()) || !finite(grid.dy()) || !(grid.dx() > 0) || !(grid.dy() > 0)
        || !finite(static_cast<Scalar>(x)) || !finite(static_cast<Scalar>(y))
        || x < grid.minx || x > grid.maxx || y < grid.miny || y > grid.maxy)
        throw std::invalid_argument("sampling coordinates or surface bounds are invalid");
    const FaultGeometry barriers(surface, faults);
    grid.faults = barriers.empty() ? nullptr : &barriers;
    PointConstraint row;
    if (!samplingConstraint(grid, x, y, row))
        throw std::invalid_argument("cannot sample on a fault or in an unresolved fault block; refine nx/ny");
    Scalar value = 0;
    for (std::size_t k = 0; k < row.count; ++k) {
        const Scalar sample = static_cast<Scalar>(surface.grid[row.terms[k].index]);
        if (!finite(sample)) throw std::invalid_argument("sampling support contains a non-finite value");
        value += row.terms[k].coefficient * sample;
    }
    const qreal result = static_cast<qreal>(value);
    if (!finite(static_cast<Scalar>(result))) throw std::runtime_error("sample does not fit in qreal");
    return result;
}

} // namespace convergent
