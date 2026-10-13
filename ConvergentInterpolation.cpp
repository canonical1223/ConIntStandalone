#include "ConvergentInterpolation.h"

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

namespace ConvergentInterpolation {
namespace {

// Конвейер: проверка входа -> уровни сетки [уточнение, Snap/Тейлор, PCG] ->
// точная проекция -> проверка результата -> запись результата.
// Публичные функции находятся в конце файла. Все вспомогательные типы
// и функции приватны; разделы ниже расположены по зависимостям.
// Порядок арифметики сохранен, в том числе в разных ветвях выборки и двух PCG.

// ============================================================================
// 1. Внутренние данные и базовые операции
// ============================================================================

class FaultGeometry;

// Фиксированные правила алгоритма; настройки пользователя хранятся в options.
constexpr std::size_t solverResidualRestartPeriod = 50;
constexpr std::size_t controlProjectionRestartPeriod = 25;
constexpr std::size_t nearestVisibleNodeRadius = 2;
constexpr std::size_t snapSearchPadding = 3;
constexpr std::size_t bilinearCornerCount = 4;
constexpr std::uint8_t curvatureXStencilBit = 1;
constexpr std::uint8_t curvatureYStencilBit = 2;
constexpr std::uint8_t curvatureMixedStencilBit = 4;

struct CanonicalPoint {
    double x{};
    double y{};
    double value{};
    double weight{};
};

// Размеры, координаты и барьеры сетки без массива значений.
struct GridGeometry {
    std::size_t nx{};
    std::size_t ny{};
    double minx{};
    double maxx{};
    double miny{};
    double maxy{};
    // Невладеющий указатель: геометрия живет дольше всех уровней расчета.
    const FaultGeometry* faults{};

    std::size_t index(std::size_t ix, std::size_t iy) const
    {
        return iy * nx + ix;
    }

    double dx() const
    {
        return (maxx - minx) / static_cast<double>(nx - 1);
    }

    double dy() const
    {
        return (maxy - miny) / static_cast<double>(ny - 1);
    }

    double x(std::size_t ix) const
    {
        return ix + 1 == nx ? maxx : minx + static_cast<double>(ix) * dx();
    }

    double y(std::size_t iy) const
    {
        return iy + 1 == ny ? maxy : miny + static_cast<double>(iy) * dy();
    }
};

// Геометрия и значения образуют полноценную сетку.
struct Grid {
    GridGeometry geometry;
    std::vector<double> values;
};

struct Derivatives {
    double gx{};
    double gy{};
    double gxx{};
    double gxy{};
    double gyy{};
};

struct StencilTerm {
    std::size_t index{};
    double coefficient{};
};

struct PointConstraint {
    std::array<StencilTerm, bilinearCornerCount> terms{};
    std::size_t count{};
    double value{};
    double weight{};
};

// Значения решения и признак сходимости для обработки лимита итераций.
struct SolverResult {
    std::vector<double> values;
    bool converged{};
};

inline bool finite(double value)
{
    return std::isfinite(value);
}

inline std::size_t checkedNodeCount(std::size_t nx, std::size_t ny)
{
    if (nx == 0 || ny > std::numeric_limits<std::size_t>::max() / nx) {
        throw std::invalid_argument("Размеры surface превышают диапазон size_t.");
    }
    return nx * ny;
}

inline double constraintValue(const PointConstraint& constraint,
                              const std::vector<double>& values)
{
    double result = 0.0;
    for (std::size_t i = 0; i < constraint.count; ++i) {
        result += constraint.terms[i].coefficient * values[constraint.terms[i].index];
    }
    return result;
}

inline GridGeometry gridGeometry(const Grid& source)
{
    return source.geometry;
}

// ============================================================================
// 2. Геометрия разломов и видимость
// ============================================================================

// Пересечение и любое касание разлома блокируют прямую видимость.
class FaultGeometry {
public:
    FaultGeometry(const Surface& surface, const std::vector<Fault>& faults);

    bool empty() const { return segments_.empty(); }
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

    // Узлы BVH хранятся в прямом порядке; escape пропускает поддерево без стека.
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
        throw std::invalid_argument(
            "Границы surface для геометрии разломов должны быть конечными и задавать положительные "
            "размеры.");
    }
    scale_ = std::max(width, height);

    for (std::size_t faultIndex = 0; faultIndex < faults.size(); ++faultIndex) {
        const auto& fault = faults[faultIndex];
        const auto rejectFault = [faultIndex] {
            throw std::invalid_argument("Разлом " + std::to_string(faultIndex) +
                                        " должен содержать хотя бы две различные вершины XY.");
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
    const std::size_t nodeIndex = index_.size();
    index_.push_back({box, begin, end - begin, 0});
    if (end - begin > maximumLeafSegments) {
        // Медиана по большей стороне сохраняет логарифмическую глубину.
        const bool splitAlongX = box.maxx - box.minx >= box.maxy - box.miny;
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(segments_.begin() + begin, segments_.begin() + middle,
                         segments_.begin() + end,
            [splitAlongX](const Segment& a, const Segment& b) {
                return splitAlongX ? a.box.minx + a.box.maxx < b.box.minx + b.box.maxx
                              : a.box.miny + a.box.maxy < b.box.miny + b.box.maxy;
            });
        index_[nodeIndex].count = 0;
        buildIndex(begin, middle);
        buildIndex(middle, end);
    }
    index_[nodeIndex].escape = index_.size();
}

FaultGeometry::XY FaultGeometry::normalize(double x, double y) const
{
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw std::invalid_argument("Координаты геометрии разломов должны быть конечными.");
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
        throw std::invalid_argument(
            "Координаты геометрии разломов выходят за допустимый диапазон после нормализации.");
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

template <class BoxPredicate, class SegmentPredicate>
bool FaultGeometry::anySegment(BoxPredicate boxMatches,
                               SegmentPredicate segmentMatches) const
{
    if (!index_.empty()) {
        std::size_t nodeIndex = 0;
        while (nodeIndex < index_.size()) {
            const IndexNode& indexNode = index_[nodeIndex];
            if (!boxMatches(indexNode.box)) {
                nodeIndex = indexNode.escape;
                continue;
            }
            for (std::size_t i = indexNode.begin; i < indexNode.begin + indexNode.count; ++i) {
                if (segmentMatches(segments_[i])) {
                    return true;
                }
            }
            ++nodeIndex;
        }
        return false;
    }
    for (const auto& segment : segments_) {
        if (segmentMatches(segment)) {
            return true;
        }
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

inline bool visibleNodes(const GridGeometry& geometry, std::size_t a, std::size_t b)
{
    return !geometry.faults || geometry.faults->visible(
        geometry.x(a % geometry.nx), geometry.y(a / geometry.nx),
        geometry.x(b % geometry.nx), geometry.y(b / geometry.nx));
}

// ============================================================================
// 3. Проверка входных данных и канонизация точек
// ============================================================================

// Проверка входа и построение многоуровневой поверхности.

void requirePositiveCount(std::size_t value, const char* parameterName)
{
    if (value == 0) {
        throw std::invalid_argument(
            std::string("Параметр ") + parameterName + " должен быть больше нуля.");
    }
}

void requirePositiveFinite(double value, const char* parameterName)
{
    if (!finite(value) || !(value > 0.0)) {
        throw std::invalid_argument(
            std::string("Параметр ") + parameterName + " должен быть конечным и положительным.");
    }
}

void requireNonNegativeFinite(double value, const char* parameterName)
{
    if (!finite(value) || value < 0.0) {
        throw std::invalid_argument(
            std::string("Параметр ") + parameterName + " должен быть конечным и неотрицательным.");
    }
}

void validateOptions(const ConvergentInterpolationOptions& options)
{
    requirePositiveCount(options.initialSnapNodes, "initialSnapNodes");
    requirePositiveCount(options.coarsestIntervals, "coarsestIntervals");
    requirePositiveCount(options.maxLevels, "maxLevels");
    requirePositiveCount(options.maxSolverIterations, "maxSolverIterations");
    if (options.enforceExactControls) {
        requirePositiveCount(options.maxControlProjectionIterations,
                             "maxControlProjectionIterations");
    }

    requireNonNegativeFinite(options.smoothness, "smoothness");
    requirePositiveFinite(options.priorWeight, "priorWeight");
    requireNonNegativeFinite(options.snapStrength, "snapStrength");
    requireNonNegativeFinite(options.finalPointStrength, "finalPointStrength");
    requirePositiveFinite(options.gaussianSigma, "gaussianSigma");
    requirePositiveFinite(options.relativeTolerance, "relativeTolerance");
    requireNonNegativeFinite(options.absoluteTolerance, "absoluteTolerance");
    requireNonNegativeFinite(options.controlTolerance, "controlTolerance");

    const double sigmaSquared = options.gaussianSigma * options.gaussianSigma;
    if (!finite(sigmaSquared) || !(sigmaSquared > 0.0)) {
        throw std::invalid_argument("Квадрат gaussianSigma должен быть конечным и положительным.");
    }
    if (options.taylorOrder < 0 || options.taylorOrder > 2) {
        throw std::invalid_argument("Параметр taylorOrder должен быть равен 0, 1 или 2.");
    }
}

// Полная проверка входной поверхности. Публичная выборка ниже проверяет
// только геометрию и фактически используемые значения, не весь массив.
void validateSurface(const Surface& surface)
{
    if (surface.nx < 2 || surface.ny < 2) {
        throw std::invalid_argument("Значения surface.nx и surface.ny должны быть не меньше 2.");
    }

    const double minx = surface.minx;
    const double maxx = surface.maxx;
    const double miny = surface.miny;
    const double maxy = surface.maxy;
    if (!finite(minx) || !finite(maxx) || !finite(miny) || !finite(maxy)
        || !(minx < maxx) || !(miny < maxy)) {
        throw std::invalid_argument(
            "Границы surface должны быть конечными и строго возрастать по каждой оси.");
    }

    const double spanX = maxx - minx;
    const double spanY = maxy - miny;
    const double dx = spanX / static_cast<double>(surface.nx - 1);
    const double dy = spanY / static_cast<double>(surface.ny - 1);
    if (!finite(spanX) || !finite(spanY) || !finite(dx) || !finite(dy)
        || !(spanX > 0) || !(spanY > 0) || !(dx > 0) || !(dy > 0)
        || !finite(dx / dy) || !finite(dy / dx)) {
        throw std::invalid_argument("Границы surface задают недопустимый шаг сетки.");
    }

    const std::size_t expected = checkedNodeCount(surface.nx, surface.ny);
    if (surface.grid.size() != expected) {
        throw std::invalid_argument(
            "Значение surface.grid.size() должно быть равно surface.nx * surface.ny.");
    }
    for (double value : surface.grid) {
        if (!finite(value)) {
            throw std::invalid_argument("Массив surface.grid должен содержать только конечные значения.");
        }
    }
}

std::vector<CanonicalPoint> validateControlPoints(
    const Surface& surface, const std::vector<Point>& points)
{
    std::vector<CanonicalPoint> result;
    result.reserve(points.size());
    for (const Point& point : points) {
        const CanonicalPoint p{
            point.x,
            point.y,
            point.value,
            point.weight};

        if (!finite(p.x) || !finite(p.y) || !finite(p.value) || !finite(p.weight)) {
            throw std::invalid_argument("Поля контрольной точки должны содержать конечные значения.");
        }
        if (p.weight < 0) {
            throw std::invalid_argument("Вес weight контрольной точки не должен быть отрицательным.");
        }
        if (p.x < surface.minx
            || p.x > surface.maxx
            || p.y < surface.miny
            || p.y > surface.maxy) {
            throw std::invalid_argument("Контрольная точка находится за границами surface.");
        }
        if (p.weight == 0) {
            continue;
        }
        result.push_back(p);
    }
    return result;
}

// Сортировка и объединение задают один воспроизводимый порядок суммирования.
std::vector<CanonicalPoint> mergeCoincidentPoints(std::vector<CanonicalPoint> controls)
{
    // Канонический порядок делает накопление воспроизводимым и не зависящим от
    // перестановки одного и того же набора входных точек.
    std::sort(controls.begin(), controls.end(), [](const CanonicalPoint& a, const CanonicalPoint& b) {
        if (a.x != b.x) {
            return a.x < b.x;
        }
        if (a.y != b.y) {
            return a.y < b.y;
        }
        if (a.value != b.value) {
            return a.value < b.value;
        }
        return a.weight < b.weight;
    });

    std::vector<CanonicalPoint> grouped;
    grouped.reserve(controls.size());
    for (std::size_t begin = 0; begin < controls.size();) {
        std::size_t end = begin + 1;
        while (end < controls.size()
               && controls[end].x == controls[begin].x
               && controls[end].y == controls[begin].y) {
            ++end;
        }
        long double weightSum = 0.0;
        long double weightedValue = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            weightSum += static_cast<long double>(controls[i].weight);
            weightedValue += static_cast<long double>(controls[i].weight)
                * static_cast<long double>(controls[i].value);
        }
        const long double combinedValue = weightedValue / weightSum;
        if (!std::isfinite(weightSum) || !std::isfinite(combinedValue)
            || weightSum > static_cast<long double>(std::numeric_limits<double>::max())
            || std::abs(combinedValue)
                > static_cast<long double>(std::numeric_limits<double>::max())) {
            throw std::invalid_argument(
                "Переполнение при объединении весов контрольных точек с совпадающими координатами.");
        }
        grouped.push_back({controls[begin].x, controls[begin].y,
                           static_cast<double>(combinedValue),
                           static_cast<double>(weightSum)});
        begin = end;
    }
    return grouped;
}

void normalizeControlWeights(std::vector<CanonicalPoint>& controls)
{
    double maximumWeight = 0.0;
    for (const CanonicalPoint& point : controls) {
        maximumWeight = std::max(maximumWeight, point.weight);
    }
    if (maximumWeight > 0) {
        for (CanonicalPoint& point : controls) {
            point.weight /= maximumWeight;
        }
    }
}

std::vector<CanonicalPoint> validateInput(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentInterpolationOptions& options)
{
    // Порядок проверок сохраняется и для случая без активных точек.
    validateOptions(options);
    validateSurface(surface);
    auto controls = mergeCoincidentPoints(validateControlPoints(surface, points));
    if (options.normalizePointWeights) {
        normalizeControlWeights(controls);
    }
    return controls;
}

// ============================================================================
// 4. Выборка поверхности, перенос и построение иерархии
// ============================================================================

// Геометрия не требует массива значений: проверки иерархии обходятся без него.
GridGeometry gridGeometry(const Surface& surface, std::size_t nx, std::size_t ny)
{
    GridGeometry result;
    result.nx = nx;
    result.ny = ny;
    result.minx = surface.minx;
    result.maxx = surface.maxx;
    result.miny = surface.miny;
    result.maxy = surface.maxy;
    return result;
}



// Координаты одной билинейной ячейки, включая замкнутую верхнюю границу.
struct Cell {
    std::size_t ix0, iy0, ix1, iy1;
    double tx, ty;
};

Cell cellAt(const GridGeometry& geometry, double x, double y)
{
    const double fx = std::clamp((x - geometry.minx) / geometry.dx(), 0.0,
                                 static_cast<double>(geometry.nx - 1));
    const double fy = std::clamp((y - geometry.miny) / geometry.dy(), 0.0,
                                 static_cast<double>(geometry.ny - 1));
    const auto ix = static_cast<std::size_t>(std::floor(fx));
    const auto iy = static_cast<std::size_t>(std::floor(fy));
    return {ix, iy, std::min(ix + 1, geometry.nx - 1), std::min(iy + 1, geometry.ny - 1),
            fx - static_cast<double>(ix), fy - static_cast<double>(iy)};
}

// Нулевые веса пропускаются, совпавшие индексы углов объединяются.
void addTerm(PointConstraint& constraint, std::size_t index, double coefficient)
{
    if (coefficient == 0) {
        return;
    }
    for (std::size_t i = 0; i < constraint.count; ++i) {
        if (constraint.terms[i].index == index) {
            constraint.terms[i].coefficient += coefficient;
            return;
        }
    }
    constraint.terms[constraint.count++] = {index, coefficient};
}

// Обычная билинейная опора; addTerm объединяет совпавшие углы на границе.
PointConstraint bilinearConstraint(const GridGeometry& geometry, const Cell& cell)
{
    const auto [ix, iy, ix1, iy1, tx, ty] = cell;
    PointConstraint corners;
    addTerm(corners, geometry.index(ix, iy), (1.0 - tx) * (1.0 - ty));
    addTerm(corners, geometry.index(ix1, iy), tx * (1.0 - ty));
    addTerm(corners, geometry.index(ix, iy1), (1.0 - tx) * ty);
    addTerm(corners, geometry.index(ix1, iy1), tx * ty);
    return corners;
}

// Выбираем видимые и взаимно совместимые углы с максимальной суммой весов.
// При равных суммах сохраняем первое подмножество в исходном порядке масок.
bool visibleCornerConstraint(const GridGeometry& geometry, double x, double y,
                             const PointConstraint& corners, PointConstraint& result)
{
    unsigned visibleCorners = 0;
    for (std::size_t k = 0; k < corners.count; ++k) {
        const auto index = corners.terms[k].index;
        if (geometry.faults->visible(
                x, y, geometry.x(index % geometry.nx), geometry.y(index / geometry.nx))) {
            visibleCorners |= 1u << k;
        }
    }
    unsigned bestSubset = 0;
    double bestWeight = 0.0;
    // Каждая пара углов встречается в нескольких из 15 подмножеств.
    // Проверяем геометрию пары только один раз, сохраняя прежний порядок
    // перебора подмножеств и сложения весов (результат выборки не меняется).
    std::array<unsigned, bilinearCornerCount> compatibleWith{};
    for (std::size_t a = 0; a < corners.count; ++a) {
        if (!(visibleCorners & (1u << a))) {
            continue;
        }
        for (std::size_t b = a + 1; b < corners.count; ++b) {
            if ((visibleCorners & (1u << b))
                && visibleNodes(geometry, corners.terms[a].index, corners.terms[b].index)) {
                compatibleWith[a] |= 1u << b;
            }
        }
    }
    for (unsigned mask = 1; mask < (1u << corners.count); ++mask) {
        if ((mask & visibleCorners) != mask) {
            continue;
        }
        bool subsetCompatible = true;
        double weight = 0.0;
        for (std::size_t a = 0; a < corners.count && subsetCompatible; ++a) {
            if (!(mask & (1u << a))) {
                continue;
            }
            weight += corners.terms[a].coefficient;
            for (std::size_t b = a + 1; b < corners.count; ++b) {
                if ((mask & (1u << b)) && !(compatibleWith[a] & (1u << b))) {
                    subsetCompatible = false;
                    break;
                }
            }
        }
        if (subsetCompatible && weight > bestWeight) {
            bestSubset = mask;
            bestWeight = weight;
        }
    }
    if (bestWeight > 0) {
        for (std::size_t k = 0; k < corners.count; ++k) {
            if (bestSubset & (1u << k)) {
                addTerm(result, corners.terms[k].index,
                                         corners.terms[k].coefficient / bestWeight);
            }
        }
        return true;
    }
    return false;
}

bool nearestVisibleConstraint(const GridGeometry& geometry, double x, double y,
                              const Cell& cell, PointConstraint& result)
{
    // Поиск ограничен двумя ячейками вокруг исходной билинейной ячейки.
    const std::size_t centerX = cell.ix0;
    const std::size_t centerY = cell.iy0;
    const double fractionalX = std::clamp((x - geometry.minx) / geometry.dx(), 0.0,
                                         static_cast<double>(geometry.nx - 1));
    const double fractionalY = std::clamp((y - geometry.miny) / geometry.dy(), 0.0,
                                         static_cast<double>(geometry.ny - 1));
    const std::size_t minX = centerX > nearestVisibleNodeRadius
        ? centerX - nearestVisibleNodeRadius : 0;
    const std::size_t minY = centerY > nearestVisibleNodeRadius
        ? centerY - nearestVisibleNodeRadius : 0;
    const std::size_t maxX = std::min(centerX + nearestVisibleNodeRadius, geometry.nx - 1);
    const std::size_t maxY = std::min(centerY + nearestVisibleNodeRadius, geometry.ny - 1);
    double bestDistanceSquared = std::numeric_limits<double>::infinity();
    std::size_t bestIndex = 0;

    for (std::size_t iy = minY; iy <= maxY; ++iy) {
        for (std::size_t ix = minX; ix <= maxX; ++ix) {
            const double offsetX = static_cast<double>(ix) - fractionalX;
            const double offsetY = static_cast<double>(iy) - fractionalY;
            const double distanceSquared = offsetX * offsetX + offsetY * offsetY;
            if (distanceSquared < bestDistanceSquared
                && geometry.faults->visible(x, y, geometry.x(ix), geometry.y(iy))) {
                bestDistanceSquared = distanceSquared;
                bestIndex = geometry.index(ix, iy);
            }
        }
    }
    if (!finite(bestDistanceSquared)) {
        return false;
    }
    addTerm(result, bestIndex, 1.0);
    return true;
}

// Один оператор опоры для привязки точек, переноса и публичной выборки.
// При неудаче result остается пустым; успешный вызов дает до четырех узлов.
bool samplingConstraint(const GridGeometry& geometry, double x, double y,
                        PointConstraint& result)
{
    result = {};
    if (geometry.faults && geometry.faults->onFault(x, y)) {
        return false;
    }
    const Cell cell = cellAt(geometry, x, y);
    const PointConstraint corners = bilinearConstraint(geometry, cell);
    if (!geometry.faults) {
        result = corners;
        return true;
    }
    return visibleCornerConstraint(geometry, x, y, corners, result)
        || nearestVisibleConstraint(geometry, x, y, cell, result);
}

double rawBilinearSample(const Grid& grid, double x, double y)
{
    const GridGeometry& geometry = grid.geometry;
    const auto [ix0, iy0, ix1, iy1, tx, ty] = cellAt(geometry, x, y);
    const double lowerLeftValue = grid.values[geometry.index(ix0, iy0)];
    const double lowerRightValue = grid.values[geometry.index(ix1, iy0)];
    const double upperLeftValue = grid.values[geometry.index(ix0, iy1)];
    const double upperRightValue = grid.values[geometry.index(ix1, iy1)];
    return (1.0 - tx) * (1.0 - ty) * lowerLeftValue
         + tx * (1.0 - ty) * lowerRightValue
         + (1.0 - tx) * ty * upperLeftValue
         + tx * ty * upperRightValue;
}

double bilinearSample(const Grid& grid, double x, double y)
{
    const GridGeometry& geometry = grid.geometry;
    if (!geometry.faults) {
        return rawBilinearSample(grid, x, y);
    }
    PointConstraint row;
    if (!samplingConstraint(geometry, x, y, row)) {
        throw std::invalid_argument(
            "Невозможно получить значение на разломе или в блоке без локальных узлов сетки; увеличьте nx и ny.");
    }
    return constraintValue(row, grid.values);
}

Grid resample(const Grid& source, std::size_t nx, std::size_t ny)
{
    Grid result;
    result.geometry = gridGeometry(source);
    const GridGeometry& sourceGeometry = source.geometry;
    GridGeometry& targetGeometry = result.geometry;
    targetGeometry.nx = nx;
    targetGeometry.ny = ny;
    result.values.assign(checkedNodeCount(nx, ny), 0.0);

    for (std::size_t iy = 0; iy < ny; ++iy) {
        const double ty = static_cast<double>(iy) / static_cast<double>(ny - 1);
        // С разломами один и тот же способ вычисления координат обязателен
        // для resample, геометрических масок и Snap: разные порядки округления
        // на больших XY могут переместить пограничный узел на другую сторону.
        const double y = sourceGeometry.faults ? targetGeometry.y(iy) : (iy + 1 == ny
            ? targetGeometry.maxy
            : targetGeometry.miny + ty * (targetGeometry.maxy - targetGeometry.miny));
        for (std::size_t ix = 0; ix < nx; ++ix) {
            const double tx = static_cast<double>(ix) / static_cast<double>(nx - 1);
            const double x = sourceGeometry.faults ? targetGeometry.x(ix) : (ix + 1 == nx
                ? targetGeometry.maxx
                : targetGeometry.minx + tx * (targetGeometry.maxx - targetGeometry.minx));
            // Узел ровно на разломе не представляет ни одну из двух сторон.
            // Его буферное значение не участвует в связях/выборках; сохраняем
            // обычный prior, чтобы grid оставался конечным и без NaN-масок.
            result.values[targetGeometry.index(ix, iy)] = sourceGeometry.faults && sourceGeometry.faults->onFault(x, y)
                ? rawBilinearSample(source, x, y) : bilinearSample(source, x, y);
        }
    }
    return result;
}

// Копирует значения surface.grid, сохраняя порядок хранения и геометрию.
Grid surfaceAsGrid(const Surface& surface)
{
    return {gridGeometry(surface, surface.nx, surface.ny), surface.grid};
}

// Целочисленное деление с округлением вверх без сложения value+divisor-1,
// которое могло бы переполнить size_t.
std::size_t ceilDivide(std::size_t value, std::size_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

std::vector<std::pair<std::size_t, std::size_t>> buildHierarchy(
    std::size_t finalNx,
    std::size_t finalNy,
    const ConvergentInterpolationOptions& options)
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

// Проверка представимости: каждый активный узел целевого
// уровня должен иметь одностороннюю опору на исходном. Проверяются именно
// соседние переходы, поскольку для 512 узлов уровни не строго вложены.
bool supportsTransfer(const GridGeometry& sourceGeometry, const GridGeometry& targetGeometry)
{
    PointConstraint row;
    for (std::size_t j = 0; j < targetGeometry.ny; ++j) {
        for (std::size_t i = 0; i < targetGeometry.nx; ++i) {
            const double x = targetGeometry.x(i), y = targetGeometry.y(j);
            if (sourceGeometry.faults->onFault(x, y)) {
                continue;
            }
            if (!samplingConstraint(sourceGeometry, x, y, row)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::pair<std::size_t, std::size_t>> faultHierarchy(
    const Surface& surface, const GridGeometry& inputGeometry,
    const std::vector<CanonicalPoint>& controls,
    const ConvergentInterpolationOptions& options)
{
    auto hierarchy = buildHierarchy(surface.nx, surface.ny, options);
    if (!inputGeometry.faults) {
        return hierarchy;
    }
    std::size_t firstSupportedLevel = 0;
    for (std::size_t level = 0; level < hierarchy.size(); ++level) {
        GridGeometry currentGeometry = gridGeometry(
            surface, hierarchy[level].first, hierarchy[level].second);
        currentGeometry.faults = inputGeometry.faults;
        PointConstraint row;
        bool supported = true;
        for (const auto& point : controls) {
            if (!samplingConstraint(currentGeometry, point.x, point.y, row)) {
                supported = false;
                break;
            }
        }
        if (supported && level + 1 < hierarchy.size()) {
            GridGeometry nextGeometry = gridGeometry(
                surface, hierarchy[level + 1].first, hierarchy[level + 1].second);
            nextGeometry.faults = inputGeometry.faults;
            supported = supportsTransfer(currentGeometry, nextGeometry);
        }
        if (supported) {
            supported = supportsTransfer(inputGeometry, currentGeometry);
        }
        if (!supported) {
            firstSupportedLevel = level + 1;
        }
    }
    if (firstSupportedLevel == hierarchy.size()) {
        throw std::invalid_argument(
            "Контрольная точка находится в блоке разломов без локальных узлов сетки; увеличьте nx и ny.");
    }
    hierarchy.erase(hierarchy.begin(), hierarchy.begin() + static_cast<std::ptrdiff_t>(firstSupportedLevel));
    return hierarchy;
}

// ============================================================================
// 5. Производные, Snap/Тейлор и точечные ограничения
// ============================================================================

double valueAt(const Grid& grid, std::size_t ix, std::size_t iy)
{
    const GridGeometry& geometry = grid.geometry;
    return grid.values[geometry.index(ix, iy)];
}

Derivatives derivativesAtNode(const Grid& grid, std::size_t ix, std::size_t iy)
{
    const GridGeometry& geometry = grid.geometry;
    const double stepX = geometry.dx();
    const double stepY = geometry.dy();
    Derivatives derivatives;

    if (geometry.faults) {
        const auto center = geometry.index(ix, iy);
        if (geometry.faults->onFault(geometry.x(ix), geometry.y(iy))) {
            return derivatives;
        }
        const auto calculateAxisDerivatives = [&](
            bool alongX, double& firstDerivative, double& secondDerivative) {
            const auto position = alongX ? ix : iy;
            const auto axisNodeCount = alongX ? geometry.nx : geometry.ny;
            const double step = alongX ? stepX : stepY;
            const auto indexAt = [&](std::size_t axisIndex) {
                return alongX ? geometry.index(axisIndex, iy) : geometry.index(ix, axisIndex);
            };
            const auto isVisibleFromCenter = [&](std::size_t axisIndex) {
                return visibleNodes(geometry, center, indexAt(axisIndex));
            };
            const auto axisValue = [&](std::size_t axisIndex) {
                return grid.values[indexAt(axisIndex)];
            };
            const double centerValue = grid.values[center];
            const bool leftVisible = position > 0 && isVisibleFromCenter(position - 1);
            const bool rightVisible = position + 1 < axisNodeCount && isVisibleFromCenter(position + 1);
            if (leftVisible && rightVisible) {
                firstDerivative = (axisValue(position + 1) - axisValue(position - 1)) / (2.0 * step);
                secondDerivative = (axisValue(position - 1) - 2.0 * centerValue
                    + axisValue(position + 1)) / (step * step);
            } else if (rightVisible) {
                firstDerivative = (axisValue(position + 1) - centerValue) / step;
                if (position + 2 < axisNodeCount && isVisibleFromCenter(position + 2)) {
                    secondDerivative = (centerValue - 2.0 * axisValue(position + 1)
                    + axisValue(position + 2)) / (step * step);
                }
            } else if (leftVisible) {
                firstDerivative = (centerValue - axisValue(position - 1)) / step;
                if (position >= 2 && isVisibleFromCenter(position - 2)) {
                    secondDerivative = (centerValue - 2.0 * axisValue(position - 1)
                    + axisValue(position - 2)) / (step * step);
                }
            }
        };
        calculateAxisDerivatives(true, derivatives.gx, derivatives.gxx);
        calculateAxisDerivatives(false, derivatives.gy, derivatives.gyy);
        const bool leftVisible = ix > 0
            && visibleNodes(geometry, center, geometry.index(ix - 1, iy));
        const bool rightVisible = ix + 1 < geometry.nx
            && visibleNodes(geometry, center, geometry.index(ix + 1, iy));
        const bool belowVisible = iy > 0
            && visibleNodes(geometry, center, geometry.index(ix, iy - 1));
        const bool aboveVisible = iy + 1 < geometry.ny
            && visibleNodes(geometry, center, geometry.index(ix, iy + 1));
        const std::size_t x0 = leftVisible ? ix - 1 : ix;
        const std::size_t x1 = rightVisible ? ix + 1 : ix;
        const std::size_t y0 = belowVisible ? iy - 1 : iy;
        const std::size_t y1 = aboveVisible ? iy + 1 : iy;
        if (x1 > x0 && y1 > y0) {
            const std::array<std::size_t, bilinearCornerCount> corners{
                geometry.index(x0, y0), geometry.index(x1, y0),
                geometry.index(x0, y1), geometry.index(x1, y1)};
            bool allCornersVisible = true;
            for (std::size_t a = 0; a < bilinearCornerCount && allCornersVisible; ++a) {
                if (!visibleNodes(geometry, center, corners[a])) {
                    allCornersVisible = false;
                }
                for (std::size_t b = a + 1; b < bilinearCornerCount && allCornersVisible; ++b) {
                    if (!visibleNodes(geometry, corners[a], corners[b])) {
                        allCornersVisible = false;
                    }
                }
            }
            if (allCornersVisible) {
                derivatives.gxy = (grid.values[corners[3]] - grid.values[corners[1]]
                    - grid.values[corners[2]] + grid.values[corners[0]])
                    / (static_cast<double>(x1 - x0) * stepX * static_cast<double>(y1 - y0) * stepY);
            }
        }
        return derivatives;
    }

    const auto calculateAxisDerivatives = [&](
        bool alongX, double& firstDerivative, double& secondDerivative) {
        const auto position = alongX ? ix : iy;
        const auto axisNodeCount = alongX ? geometry.nx : geometry.ny;
        const double step = alongX ? stepX : stepY;
        const auto axisValue = [&](std::size_t axisIndex) {
            return alongX ? valueAt(grid, axisIndex, iy) : valueAt(grid, ix, axisIndex);
        };
        if (position == 0) {
            firstDerivative = (axisValue(1) - axisValue(0)) / step;
        } else if (position + 1 == axisNodeCount) {
            firstDerivative = (axisValue(position) - axisValue(position - 1)) / step;
        } else {
            firstDerivative = (axisValue(position + 1) - axisValue(position - 1)) / (2.0 * step);
        }
        if (axisNodeCount >= 3) {
            if (position == 0) {
                secondDerivative = (axisValue(0) - 2.0 * axisValue(1) + axisValue(2))
                    / (step * step);
            } else if (position + 1 == axisNodeCount) {
                secondDerivative = (axisValue(position) - 2.0 * axisValue(position - 1)
                    + axisValue(position - 2)) / (step * step);
            } else {
                secondDerivative = (axisValue(position - 1) - 2.0 * axisValue(position)
                    + axisValue(position + 1)) / (step * step);
            }
        }
    };
    calculateAxisDerivatives(true, derivatives.gx, derivatives.gxx);
    calculateAxisDerivatives(false, derivatives.gy, derivatives.gyy);

    const std::size_t ix0 = ix == 0 ? 0 : ix - 1;
    const std::size_t ix1 = ix + 1 < geometry.nx ? ix + 1 : ix;
    const std::size_t iy0 = iy == 0 ? 0 : iy - 1;
    const std::size_t iy1 = iy + 1 < geometry.ny ? iy + 1 : iy;
    const double spanX = static_cast<double>(ix1 - ix0) * stepX;
    const double spanY = static_cast<double>(iy1 - iy0) * stepY;
    if (spanX > 0 && spanY > 0) {
        derivatives.gxy = (valueAt(grid, ix1, iy1) - valueAt(grid, ix1, iy0)
            - valueAt(grid, ix0, iy1) + valueAt(grid, ix0, iy0))
            / (spanX * spanY);
    }
    return derivatives;
}

Derivatives derivativesAt(const Grid& grid, double x, double y)
{
    const GridGeometry& geometry = grid.geometry;
    if (geometry.faults) {
        PointConstraint row;
        if (!samplingConstraint(geometry, x, y, row)) {
            throw std::invalid_argument(
                "Для производных Тейлора нет видимых опорных узлов сетки; увеличьте nx и ny.");
        }
        Derivatives result;
        for (std::size_t k = 0; k < row.count; ++k) {
            const auto& term = row.terms[k];
            const Derivatives nodeDerivatives = derivativesAtNode(
                grid, term.index % geometry.nx, term.index / geometry.nx);
            result.gx += term.coefficient * nodeDerivatives.gx;
            result.gy += term.coefficient * nodeDerivatives.gy;
            result.gxx += term.coefficient * nodeDerivatives.gxx;
            result.gxy += term.coefficient * nodeDerivatives.gxy;
            result.gyy += term.coefficient * nodeDerivatives.gyy;
        }
        return result;
    }
    const auto [ix0, iy0, ix1, iy1, tx, ty] = cellAt(geometry, x, y);

    const Derivatives lowerLeftDerivatives = derivativesAtNode(grid, ix0, iy0);
    const Derivatives lowerRightDerivatives = derivativesAtNode(grid, ix1, iy0);
    const Derivatives upperLeftDerivatives = derivativesAtNode(grid, ix0, iy1);
    const Derivatives upperRightDerivatives = derivativesAtNode(grid, ix1, iy1);
    // В C++17 tx и ty из структурированной привязки захватываются по значению.
    const auto blend = [&, tx = tx, ty = ty](double Derivatives::*member) {
        return (1.0 - tx) * (1.0 - ty) * lowerLeftDerivatives.*member
             + tx * (1.0 - ty) * lowerRightDerivatives.*member
             + (1.0 - tx) * ty * upperLeftDerivatives.*member
             + tx * ty * upperRightDerivatives.*member;
    };
    return {blend(&Derivatives::gx), blend(&Derivatives::gy),
            blend(&Derivatives::gxx), blend(&Derivatives::gxy),
            blend(&Derivatives::gyy)};
}

struct CandidateNode {
    std::size_t index{};
    std::size_t ix{};
    std::size_t iy{};
    double distanceSquared{};
};

std::vector<CandidateNode> nearestNodes(
    const GridGeometry& geometry, double x, double y, std::size_t requested)
{
    requested = std::min(requested, checkedNodeCount(geometry.nx, geometry.ny));
    const double fx = (x - geometry.minx) / geometry.dx();
    const double fy = (y - geometry.miny) / geometry.dy();
    const std::size_t centerX = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fx, 0.0, static_cast<double>(geometry.nx - 1)))), geometry.nx - 1);
    const std::size_t centerY = std::min(
        static_cast<std::size_t>(std::floor(std::clamp(
            fy, 0.0, static_cast<double>(geometry.ny - 1)))), geometry.ny - 1);
    const std::size_t radius = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(requested)))) + snapSearchPadding;
    const std::size_t minX = centerX > radius ? centerX - radius : 0;
    const std::size_t minY = centerY > radius ? centerY - radius : 0;
    const std::size_t maxX = std::min(centerX + radius, geometry.nx - 1);
    const std::size_t maxY = std::min(centerY + radius, geometry.ny - 1);

    std::vector<CandidateNode> candidates;
    const auto collect = [&](std::size_t x0, std::size_t y0,
                             std::size_t x1, std::size_t y1) {
        candidates.reserve((x1 - x0 + 1) * (y1 - y0 + 1));
        for (std::size_t iy = y0; iy <= y1; ++iy) {
            for (std::size_t ix = x0; ix <= x1; ++ix) {
                if (geometry.faults && !geometry.faults->visible(x, y, geometry.x(ix), geometry.y(iy))) {
                    continue;
                }
                const double dxCells = static_cast<double>(ix) - fx;
                const double dyCells = static_cast<double>(iy) - fy;
                candidates.push_back({geometry.index(ix, iy), ix, iy,
                                      dxCells * dxCells + dyCells * dyCells});
            }
        }
    };
    collect(minX, minY, maxX, maxY);

    // Для очень тонкой сетки или точки у края локального квадрата может быть
    // недостаточно. Редкий полный просмотр сохраняет точный результат именно
    // поиска ближайших узлов, независимо от точной привязки значений.
    if (candidates.size() < requested && !geometry.faults) {
        candidates.clear();
        collect(0, 0, geometry.nx - 1, geometry.ny - 1);
    }

    // Нужны только первые requested кандидатов. Частичная сортировка
    // сохраняет тот же упорядоченный префикс, но не сортирует
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
        throw std::invalid_argument("Для контрольной точки нет видимых узлов Snap; увеличьте nx и ny.");
    }
    // Барьер может оставить меньше узлов, чем номинальное расписание Snap.
    // Используем доступные, не расширяя связь через разлом ради нужного числа.
    candidates.resize(selected);
    return candidates;
}

std::size_t snapNodeCount(
    const GridGeometry& geometry,
    const GridGeometry& coarsest,
    bool finalLevel,
    const ConvergentInterpolationOptions& options)
{
    if (finalLevel) {
        return 1;
    }
    const std::size_t cappedInitial = std::min(
        options.initialSnapNodes, checkedNodeCount(geometry.nx, geometry.ny));
    const long double scaleSquared =
        static_cast<long double>(geometry.dx() / coarsest.dx())
        * static_cast<long double>(geometry.dy() / coarsest.dy());
    const long double scaled = static_cast<long double>(cappedInitial)
        * std::sqrt(scaleSquared);
    if (!std::isfinite(scaled) || scaled < 0
        || scaled > static_cast<long double>(checkedNodeCount(geometry.nx, geometry.ny))) {
        throw std::invalid_argument(
            "Невозможно масштабировать initialSnapNodes без выхода за допустимый диапазон.");
    }
    return std::max<std::size_t>(1,
        static_cast<std::size_t>(std::floor(scaled + 0.5L)));
}

// Ненормированные гауссовы веса и их сумма сохраняют порядок вычисления штрафов.
struct SnapKernel {
    std::vector<double> values;
    double sum{};
};

SnapKernel makeSnapKernel(const std::vector<CandidateNode>& nodes,
                          double sigmaSquared, bool hasFaults)
{
    // nearestNodes гарантирует непустой список, упорядоченный по расстоянию.
    SnapKernel kernel;
    kernel.values.resize(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        // При разломах сдвиг расстояний предотвращает потерю весов из-за округления
        // к нулю; при нормировке этот общий множитель сокращается.
        const double shiftedDistanceSquared = nodes[i].distanceSquared
            - (hasFaults ? nodes.front().distanceSquared : 0.0);
        kernel.values[i] = std::exp(-shiftedDistanceSquared / (2.0 * sigmaSquared));
        kernel.sum += kernel.values[i];
    }
    if (!(kernel.sum > 0.0) || !finite(kernel.sum)) {
        throw std::runtime_error("Не удалось нормализовать весовое ядро расстояний Snap.");
    }
    return kernel;
}

double projectPointToNode(
    const GridGeometry& geometry,
    const CanonicalPoint& point,
    const CandidateNode& node,
    const Derivatives& derivatives,
    int taylorOrder)
{
    // Способ вычисления координат соответствует переносу поверхности с разломами.
    const double nodeX = geometry.faults ? geometry.x(node.ix)
        : geometry.minx + static_cast<double>(node.ix) * geometry.dx();
    const double nodeY = geometry.faults ? geometry.y(node.iy)
        : geometry.miny + static_cast<double>(node.iy) * geometry.dy();
    const double deltaX = nodeX - point.x;
    const double deltaY = nodeY - point.y;
    double projectedValue = point.value;
    if (taylorOrder >= 1) {
        projectedValue += derivatives.gx * deltaX + derivatives.gy * deltaY;
    }
    if (taylorOrder >= 2) {
        projectedValue += 0.5 * (derivatives.gxx * deltaX * deltaX
            + 2.0 * derivatives.gxy * deltaX * deltaY
            + derivatives.gyy * deltaY * deltaY);
    }
    if (!finite(projectedValue)) {
        throw std::runtime_error("Проекция Тейлора дала неконечное значение.");
    }
    return projectedValue;
}

void accumulateSnapConstraint(
    std::size_t nodeIndex, double weight, double projectedValue,
    std::vector<double>& accumulatedWeight, std::vector<double>& accumulatedValue)
{
    const double nextWeight = accumulatedWeight[nodeIndex] + weight;
    const double nextValue = accumulatedValue[nodeIndex] + weight * projectedValue;
    if (!finite(weight) || !finite(nextWeight) || !finite(nextValue)) {
        throw std::runtime_error("Переполнение при накоплении ограничений Snap.");
    }
    accumulatedWeight[nodeIndex] = nextWeight;
    accumulatedValue[nodeIndex] = nextValue;
}

void addSnapConstraints(
    const Grid& prior,
    const std::vector<CanonicalPoint>& points,
    std::size_t requestedNodes,
    const ConvergentInterpolationOptions& options,
    std::vector<double>& diagonal,
    std::vector<double>& rhs)
{
    std::vector<double> accumulatedWeight(prior.values.size(), 0.0);
    std::vector<double> accumulatedValue(prior.values.size(), 0.0);
    const double sigmaSquared = options.gaussianSigma * options.gaussianSigma;

    for (const CanonicalPoint& point : points) {
        const Derivatives derivatives = derivativesAt(prior, point.x, point.y);
        const std::vector<CandidateNode> nodes = nearestNodes(
            prior.geometry, point.x, point.y, requestedNodes);
        const SnapKernel kernel = makeSnapKernel(
            nodes, sigmaSquared, prior.geometry.faults != nullptr);

        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const CandidateNode& node = nodes[i];
            const double projectedValue = projectPointToNode(
                prior.geometry, point, node, derivatives, options.taylorOrder);
            const double weight = point.weight * kernel.values[i] / kernel.sum;
            accumulateSnapConstraint(node.index, weight, projectedValue,
                                     accumulatedWeight, accumulatedValue);
        }
    }

    const double strength = options.snapStrength;
    for (std::size_t i = 0; i < diagonal.size(); ++i) {
        diagonal[i] += strength * accumulatedWeight[i];
        rhs[i] += strength * accumulatedValue[i];
        if (!finite(diagonal[i]) || !finite(rhs[i])) {
            throw std::runtime_error("Переполнение при масштабировании ограничений Snap.");
        }
    }
}

// Единый оператор C для мягких и точных ограничений и публичной выборки.
std::vector<PointConstraint> makePointConstraints(
    const GridGeometry& geometry, const std::vector<CanonicalPoint>& points)
{
    std::vector<PointConstraint> result;
    result.reserve(points.size());
    for (const CanonicalPoint& point : points) {
        PointConstraint constraint;
        if (!samplingConstraint(geometry, point.x, point.y, constraint)) {
            throw std::invalid_argument(
                "Для контрольной точки нет опорных узлов интерполяции с учётом разломов; увеличьте nx и "
                "ny.");
        }
        constraint.value = point.value;
        constraint.weight = point.weight;
        result.push_back(constraint);
    }
    return result;
}

// Максимум |C*u-d| по каноническим точкам для проверки точной привязки.
// При наличии разломов C совпадает с выборкой sampleSurface с учётом разломов.
double maxControlError(const Grid& grid, const std::vector<CanonicalPoint>& points)
{
    double result = 0.0;
    for (const CanonicalPoint& point : points) {
        result = std::max(result, std::abs(bilinearSample(grid, point.x, point.y) - point.value));
    }
    return result;
}

// ============================================================================
// 6. Кривизна, система уровня и основной PCG
// ============================================================================

// Удаляем целую строку B при пересечении разлома любой парой ее узлов.
// Это сохраняет симметрию и положительную полуопределенность B^T*B.
std::vector<std::uint8_t> faultCurvatureMask(const GridGeometry& geometry)
{
    if (!geometry.faults) {
        return {};
    }
    std::vector<std::uint8_t> mask(checkedNodeCount(geometry.nx, geometry.ny), 0);
    const auto allowed = [&](std::initializer_list<std::size_t> nodes) {
        for (auto a = nodes.begin(); a != nodes.end(); ++a) {
            for (auto b = a + 1; b != nodes.end(); ++b) {
                if (!visibleNodes(geometry, *a, *b)) {
                    return false;
                }
            }
        }
        return true;
    };
    for (std::size_t j = 0; j < geometry.ny; ++j) {
        for (std::size_t i = 0; i < geometry.nx; ++i) {
            const auto center = geometry.index(i, j);
            if (i > 0 && i + 1 < geometry.nx
                && allowed({geometry.index(i - 1, j), center, geometry.index(i + 1, j)})) {
                mask[center] |= curvatureXStencilBit;
            }
            if (j > 0 && j + 1 < geometry.ny
                && allowed({geometry.index(i, j - 1), center, geometry.index(i, j + 1)})) {
                mask[center] |= curvatureYStencilBit;
            }
            if (i > 0 && i + 1 < geometry.nx && j > 0 && j + 1 < geometry.ny
                && allowed({geometry.index(i + 1, j + 1), geometry.index(i + 1, j - 1),
                            geometry.index(i - 1, j + 1), geometry.index(i - 1, j - 1)})) {
                mask[center] |= curvatureMixedStencilBit;
            }
        }
    }
    return mask;
}

// Общий обход гарантирует одинаковые граничные строки и маску у K и diag(K).
// Порядок накопления сохранен: X, Y, смешанная производная; внутри — по строкам.
template <class Axial, class Mixed>
void visitCurvatureStencils(const GridGeometry& geometry, const std::vector<std::uint8_t>* mask,
                            const Axial& axial, const Mixed& mixed)
{
    const double xCurvatureScale = geometry.dy() / geometry.dx();
    const double yCurvatureScale = geometry.dx() / geometry.dy();
    if (geometry.nx >= 3) {
        for (std::size_t iy = 0; iy < geometry.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < geometry.nx; ++ix) {
                if (mask && !((*mask)[geometry.index(ix, iy)] & curvatureXStencilBit)) {
                    continue;
                }
                axial(geometry.index(ix - 1, iy), geometry.index(ix, iy),
                      geometry.index(ix + 1, iy), xCurvatureScale);
            }
        }
    }
    if (geometry.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < geometry.ny; ++iy) {
            for (std::size_t ix = 0; ix < geometry.nx; ++ix) {
                if (mask && !((*mask)[geometry.index(ix, iy)] & curvatureYStencilBit)) {
                    continue;
                }
                axial(geometry.index(ix, iy - 1), geometry.index(ix, iy),
                      geometry.index(ix, iy + 1), yCurvatureScale);
            }
        }
    }
    if (geometry.nx >= 3 && geometry.ny >= 3) {
        for (std::size_t iy = 1; iy + 1 < geometry.ny; ++iy) {
            for (std::size_t ix = 1; ix + 1 < geometry.nx; ++ix) {
                if (mask && !((*mask)[geometry.index(ix, iy)] & curvatureMixedStencilBit)) {
                    continue;
                }
                mixed(geometry.index(ix + 1, iy + 1), geometry.index(ix + 1, iy - 1),
                      geometry.index(ix - 1, iy + 1), geometry.index(ix - 1, iy - 1));
            }
        }
    }
}

// Накапливает K*input, K=smoothness*B^T*B, не очищая output.
// Накопление выполняется последовательно: соседние строки пишут в общие узлы.
void applyCurvature(
    const GridGeometry& geometry,
    const std::vector<double>& input,
    std::vector<double>& output,
    double smoothness,
    const std::vector<std::uint8_t>* mask = nullptr)
{
    if (smoothness == 0) {
        return;
    }
    visitCurvatureStencils(geometry, mask,
        [&](std::size_t a, std::size_t b, std::size_t c, double scale) {
            const double curvatureDifference = scale * (input[a] - 2.0 * input[b] + input[c]);
            const double weightedDifference = smoothness * scale * curvatureDifference;
            output[a] += weightedDifference;
            output[b] -= 2.0 * weightedDifference;
            output[c] += weightedDifference;
        },
        [&](std::size_t pp, std::size_t pm, std::size_t mp, std::size_t mm) {
            constexpr double mixedDerivativeCoefficient = 0.25;
            const double curvatureDifference = mixedDerivativeCoefficient * (input[pp] - input[pm] - input[mp] + input[mm]);
            const double weightedDifference = 2.0 * smoothness * mixedDerivativeCoefficient * curvatureDifference;
            output[pp] += weightedDifference;
            output[pm] -= weightedDifference;
            output[mp] -= weightedDifference;
            output[mm] += weightedDifference;
        });
}

// Точная диагональ K для предобуславливателя Якоби.
std::vector<double> curvatureDiagonal(const GridGeometry& geometry, double smoothness,
                                     const std::vector<std::uint8_t>* mask = nullptr)
{
    std::vector<double> diagonal(checkedNodeCount(geometry.nx, geometry.ny), 0.0);
    if (smoothness == 0) {
        return diagonal;
    }
    visitCurvatureStencils(geometry, mask,
        [&](std::size_t a, std::size_t b, std::size_t c, double scale) {
            const double factor = smoothness * scale * scale;
            diagonal[a] += factor;
            diagonal[b] += 4.0 * factor;
            diagonal[c] += factor;
        },
        [&](std::size_t pp, std::size_t pm, std::size_t mp, std::size_t mm) {
            constexpr double coefficientSquared = 1.0 / 16.0;
            const double factor = 2.0 * smoothness * coefficientSquared;
            diagonal[pp] += factor;
            diagonal[pm] += factor;
            diagonal[mp] += factor;
            diagonal[mm] += factor;
        });
    return diagonal;
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    // Последовательное накопление сохраняет округления и число PCG-итераций.
    long double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    }
    return static_cast<double>(result);
}

double euclideanNorm(const std::vector<double>& values)
{
    return std::sqrt(std::max(0.0, dot(values, values)));
}

// Система одного уровня: сборка правой части и диагонали, затем применение A.
// Ссылки действуют только в пределах solveLevel; маска разломов строится один раз.
class ConvergentInterpolationLevelSystem {
public:
    ConvergentInterpolationLevelSystem(
        const GridGeometry& geometry,
        const std::vector<double>& prior,
        const std::vector<double>& extraDiagonal,
        const std::vector<double>& extraRhs,
        const std::vector<PointConstraint>& finalConstraints,
        const ConvergentInterpolationOptions& options)
        : rhs(prior.size()),
          geometry_(geometry), extraDiagonal_(extraDiagonal),
          constraints_(finalConstraints),
          priorWeight_(options.priorWeight),
          smoothness_(options.smoothness),
          finalStrength_(options.finalPointStrength),
          maskStorage_(faultCurvatureMask(geometry))
    {
        const auto* mask = geometry_.faults ? &maskStorage_ : nullptr;
        diagonal = curvatureDiagonal(geometry_, smoothness_, mask);
        assemblePriorAndSnap(prior, extraRhs);
        assemblePointConstraints();
    }

    // A = K + priorWeight*I + diag(Snap) + sum(rho*w*c*c^T).
    void applyA(const std::vector<double>& input, std::vector<double>& output) const
    {
        const auto* mask = geometry_.faults ? &maskStorage_ : nullptr;
        std::fill(output.begin(), output.end(), 0.0);
        applyCurvature(geometry_, input, output, smoothness_, mask);
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            output[i] += (priorWeight_ + extraDiagonal_[i]) * input[i];
        }
        for (const PointConstraint& constraint : constraints_) {
            const double strength = finalStrength_ * constraint.weight;
            const double projected = constraintValue(constraint, input);
            for (std::size_t i = 0; i < constraint.count; ++i) {
                const StencilTerm& term = constraint.terms[i];
                output[term.index] += strength * term.coefficient * projected;
            }
        }
    }

    std::vector<double> rhs;
    std::vector<double> diagonal;

private:
    void assemblePriorAndSnap(const std::vector<double>& prior,
                              const std::vector<double>& extraRhs)
    {
        const std::size_t size = prior.size();
        for (std::size_t i = 0; i < size; ++i) {
            diagonal[i] += priorWeight_ + extraDiagonal_[i];
            rhs[i] = priorWeight_ * prior[i] + extraRhs[i];
            if (!finite(diagonal[i]) || !(diagonal[i] > 0) || !finite(rhs[i])) {
                throw std::runtime_error("Коэффициенты системы уровня должны быть конечными.");
            }
        }
    }

    void assemblePointConstraints()
    {
        for (const PointConstraint& constraint : constraints_) {
            const double strength = finalStrength_ * constraint.weight;
            if (!finite(strength)) {
                throw std::runtime_error("Переполнение при вычислении силы ограничения контрольной точки.");
            }
            for (std::size_t a = 0; a < constraint.count; ++a) {
                const StencilTerm& term = constraint.terms[a];
                rhs[term.index] += strength * term.coefficient * constraint.value;
                diagonal[term.index] += strength * term.coefficient * term.coefficient;
                if (!finite(rhs[term.index]) || !finite(diagonal[term.index])) {
                    throw std::runtime_error(
                        "Переполнение при вычислении коэффициентов ограничений контрольных точек.");
                }
            }
        }
    }

    const GridGeometry& geometry_;
    const std::vector<double>& extraDiagonal_;
    const std::vector<PointConstraint>& constraints_;
    double priorWeight_;
    double smoothness_;
    double finalStrength_;
    std::vector<std::uint8_t> maskStorage_;
};

// PCG отвечает только за итерации; устройство оператора скрыто в системе.
// priorWeight > 0 делает A положительно определенной; prior задаёт начальное решение.
SolverResult solveLevel(
    const GridGeometry& geometry,
    const std::vector<double>& prior,
    const std::vector<double>& extraDiagonal,
    const std::vector<double>& extraRhs,
    const std::vector<PointConstraint>& finalConstraints,
    const ConvergentInterpolationOptions& options)
{
    const ConvergentInterpolationLevelSystem system(
        geometry, prior, extraDiagonal, extraRhs, finalConstraints, options);
    const std::size_t size = prior.size();
    const auto& rhs = system.rhs;
    const auto& diagonal = system.diagonal;
    std::vector<double> solutionValues = prior;
    std::vector<double> matrixTimesSolution(size);
    std::vector<double> residual(size);
    std::vector<double> preconditionedResidual(size);
    std::vector<double> direction(size);
    std::vector<double> matrixTimesDirection(size);
    system.applyA(solutionValues, matrixTimesSolution);
    for (std::size_t i = 0; i < size; ++i) {
        residual[i] = rhs[i] - matrixTimesSolution[i];
        preconditionedResidual[i] = residual[i] / diagonal[i];
    }
    direction = preconditionedResidual;
    double residualPreconditionedDot = dot(residual, preconditionedResidual);
    const double rhsNorm = euclideanNorm(rhs);
    const double initialMatrixProductNorm = euclideanNorm(matrixTimesSolution);
    const double residualScale = std::max(rhsNorm, initialMatrixProductNorm);
    const double tolerance = options.relativeTolerance;
    const double absoluteTolerance = options.absoluteTolerance;
    // ||r|| <= absTol + relTol*max(||b||, ||A*x0||); масштаб фиксирован.
    const double convergenceThreshold = absoluteTolerance + tolerance * residualScale;
    if (!finite(residualScale) || !finite(convergenceThreshold)) {
        throw std::runtime_error("Масштаб невязки PCG должен быть конечным.");
    }
    double residualNorm = euclideanNorm(residual);

    SolverResult result;
    result.converged = residualNorm <= convergenceThreshold;
    if (result.converged) {
        result.values = std::move(solutionValues);
        return result;
    }

    for (std::size_t iteration = 0; iteration < options.maxSolverIterations; ++iteration) {
        system.applyA(direction, matrixTimesDirection);
        const double denominator = dot(direction, matrixTimesDirection);
        if (!(denominator > 0) || !finite(denominator) || !finite(residualPreconditionedDot)) {
            throw std::runtime_error("Нарушены условия работы PCG при решении бигармонической системы.");
        }
        const double alpha = residualPreconditionedDot / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error("Шаг PCG имеет неконечное значение.");
        }
        for (std::size_t i = 0; i < size; ++i) {
            solutionValues[i] += alpha * direction[i];
            residual[i] -= alpha * matrixTimesDirection[i];
        }

        // Каждые 50 шагов восстанавливаем истинную невязку и направление.
        const bool restartDirection = (iteration + 1) % solverResidualRestartPeriod == 0;
        if (restartDirection) {
            system.applyA(solutionValues, matrixTimesSolution);
            for (std::size_t i = 0; i < size; ++i) {
                residual[i] = rhs[i] - matrixTimesSolution[i];
            }
        }
        residualNorm = euclideanNorm(residual);
        if (residualNorm <= convergenceThreshold) {
            result.values = std::move(solutionValues);
            result.converged = true;
            return result;
        }

        for (std::size_t i = 0; i < size; ++i) {
            preconditionedResidual[i] = residual[i] / diagonal[i];
        }
        const double nextResidualPreconditionedDot = dot(residual, preconditionedResidual);
        if (!finite(nextResidualPreconditionedDot)) {
            throw std::runtime_error("Невязка PCG имеет неконечное значение.");
        }
        if (restartDirection) {
            direction = preconditionedResidual;
        } else {
            const double beta = nextResidualPreconditionedDot / residualPreconditionedDot;
            for (std::size_t i = 0; i < size; ++i) {
                direction[i] = preconditionedResidual[i] + beta * direction[i];
            }
        }
        residualPreconditionedDot = nextResidualPreconditionedDot;
    }

    result.values = std::move(solutionValues);
    result.converged = false;
    return result;
}

// ============================================================================
// 7. Точная проекция на контрольные точки
// ============================================================================

double maximumAbsolute(const std::vector<double>& values)
{
    double result = 0.0;
    for (double value : values) {
        if (!finite(value)) {
            throw std::runtime_error("Невязка проекции на контрольные точки должна быть конечной.");
        }
        result = std::max(result, std::abs(value));
    }
    return result;
}

double effectiveControlTolerance(
    const std::vector<PointConstraint>& constraints,
    const ConvergentInterpolationOptions& options)
{
    double valueScale = 1.0;
    for (const PointConstraint& constraint : constraints) {
        valueScale = std::max(valueScale, std::abs(constraint.value));
    }
    // Допуск учитывает машинную точность double и масштаб значений.
    const double roundingFloor = 64.0
        * std::numeric_limits<double>::epsilon() * valueScale;
    return std::max(options.controlTolerance, roundingFloor);
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

    void apply(const std::vector<double>& input, std::vector<double>& output)
    {
        scatter(input);
        for (std::size_t i = 0; i < rows_.size(); ++i) {
            output[i] = constraintValue(rows_[i], nodeWork_);
            if (!finite(output[i])) {
                throw std::runtime_error(
                    "Переполнение при вычислении произведения матрицы Грама ограничений контрольных точек "
                    "на вектор.");
            }
        }
    }

    void addCorrection(const std::vector<double>& multiplier, std::vector<double>& values)
    {
        scatter(multiplier);
        for (std::size_t i = 0; i < globalIndices_.size(); ++i) {
            values[globalIndices_[i]] += nodeWork_[i];
        }
        for (double value : values) {
            if (!finite(value)) {
                throw std::runtime_error(
                    "Точная коррекция по контрольным точкам дала неконечное значение в узле сетки.");
            }
        }
    }

private:
    void scatter(const std::vector<double>& input)
    {
        // Строки имеют общие узлы; накопление должно оставаться последовательным.
        std::fill(nodeWork_.begin(), nodeWork_.end(), 0.0);
        for (std::size_t i = 0; i < rows_.size(); ++i) {
            for (std::size_t t = 0; t < rows_[i].count; ++t) {
                const StencilTerm& term = rows_[i].terms[t];
                nodeWork_[term.index] += term.coefficient * input[i];
            }
        }
    }

    std::vector<PointConstraint> rows_;
    std::vector<std::size_t> globalIndices_;
    std::vector<double> nodeWork_;
};

// Евклидова проекция: G*lambda=d-C*u, delta=C^T*lambda. Меняет только узлы C.
// Это отдельный PCG для матрицы Грама с максимумом невязки; веса здесь не участвуют.
void projectOntoPointConstraints(
    Grid& grid,
    const std::vector<PointConstraint>& constraints,
    const ConvergentInterpolationOptions& options)
{
    if (constraints.empty()) {
        return;
    }

    const std::size_t constraintCount = constraints.size();
    const double tolerance = effectiveControlTolerance(constraints, options);
    std::vector<double> rhs(constraintCount), residual(constraintCount), inverseDiagonal(constraintCount);
    for (std::size_t i = 0; i < constraintCount; ++i) {
        rhs[i] = constraints[i].value
            - constraintValue(constraints[i], grid.values);
        double diagonal = 0.0;
        for (std::size_t termIndex = 0;
             termIndex < constraints[i].count; ++termIndex) {
            const double coefficient = constraints[i].terms[termIndex].coefficient;
            diagonal += coefficient * coefficient;
        }
        if (!finite(rhs[i]) || !finite(diagonal) || !(diagonal > 0)) {
            throw std::runtime_error("Недопустимое точное ограничение контрольной точки.");
        }
        inverseDiagonal[i] = 1.0 / diagonal;
    }

    if (maximumAbsolute(rhs) <= tolerance) {
        return;
    }

    std::vector<double> multiplier(constraintCount, 0.0);
    residual = rhs;
    std::vector<double> preconditioned(constraintCount), direction(constraintCount), gramDirection(constraintCount);
    std::vector<double> gramMultiplier(constraintCount);
    ConstraintGramOperator gram(constraints);
    for (std::size_t i = 0; i < constraintCount; ++i) {
        preconditioned[i] = residual[i] * inverseDiagonal[i];
    }
    direction = preconditioned;
    double residualPreconditioned = dot(residual, preconditioned);

    bool converged = false;
    for (std::size_t iteration = 0;
         iteration < options.maxControlProjectionIterations; ++iteration) {
        gram.apply(direction, gramDirection);
        const double denominator = dot(direction, gramDirection);
        if (!(denominator > 0) || !finite(denominator)
            || !(residualPreconditioned > 0)
            || !finite(residualPreconditioned)) {
            throw std::runtime_error(
                "Уравнения точной привязки контрольных точек зависимы или несовместимы при данном разрешении сетки.");
        }
        const double alpha = residualPreconditioned / denominator;
        if (!finite(alpha)) {
            throw std::runtime_error(
                "Шаг проекции для точной привязки контрольных точек имеет неконечное значение.");
        }
        for (std::size_t i = 0; i < constraintCount; ++i) {
            multiplier[i] += alpha * direction[i];
            residual[i] -= alpha * gramDirection[i];
        }

        // Истинная невязка проверяется каждые 25 шагов и перед успешным выходом.
        const bool restartDirection =
            (iteration + 1) % controlProjectionRestartPeriod == 0;
        double residualError = restartDirection ? 0.0 : maximumAbsolute(residual);
        if (restartDirection || residualError <= tolerance) {
            gram.apply(multiplier, gramMultiplier);
            for (std::size_t i = 0; i < constraintCount; ++i) {
                residual[i] = rhs[i] - gramMultiplier[i];
            }
            residualError = maximumAbsolute(residual);
        }

        if (residualError <= tolerance) {
            converged = true;
            break;
        }

        for (std::size_t i = 0; i < constraintCount; ++i) {
            preconditioned[i] = residual[i] * inverseDiagonal[i];
        }
        const double nextResidualPreconditioned = dot(residual, preconditioned);
        if (!(nextResidualPreconditioned > 0)
            || !finite(nextResidualPreconditioned)) {
            throw std::runtime_error(
                "Уравнения точной привязки контрольных точек зависимы или несовместимы при данном разрешении сетки.");
        }
        if (restartDirection) {
            direction = preconditioned;
        } else {
            const double beta = nextResidualPreconditioned / residualPreconditioned;
            for (std::size_t i = 0; i < constraintCount; ++i) {
                direction[i] = preconditioned[i] + beta * direction[i];
            }
        }
        residualPreconditioned = nextResidualPreconditioned;
    }

    if (!converged) {
        throw std::runtime_error(
            "Точная привязка контрольных точек не сошлась; ограничения могут быть несовместимы "
            "при данном разрешении сетки.");
    }

    gram.addCorrection(multiplier, grid.values);

    // Независимая проверка ограничений после изменения узлов сетки.
    double maxError = 0.0;
    for (const PointConstraint& constraint : constraints) {
        maxError = std::max(maxError,
            std::abs(constraintValue(constraint, grid.values) - constraint.value));
    }
    if (maxError > tolerance) {
        throw std::runtime_error(
            "Точная коррекция по контрольным точкам потеряла точность при обновлении сетки.");
    }
}

// ============================================================================
// 8. Этапы полного расчета и подготовка результата
// ============================================================================

// Решение и опора точек хранятся вместе: строки C не зависят от значений grid.
// Опора финального уровня повторно используется точной проекцией.
struct ConvergentInterpolationSolution {
    Grid grid;
    std::vector<PointConstraint> constraints;
};

ConvergentInterpolationSolution solveConvergentInterpolationHierarchy(
    const Surface& surface,
    const Grid& input,
    const std::vector<CanonicalPoint>& controls,
    const ConvergentInterpolationOptions& options)
{
    // Начальный prior — входная поверхность; далее prior — предыдущий уровень.
    const auto hierarchy = faultHierarchy(surface, input.geometry, controls, options);
    const bool useInputGrid = input.geometry.faults
        && hierarchy.front() == std::make_pair(surface.nx, surface.ny);
    Grid coarsest = useInputGrid
        ? input : resample(input, hierarchy.front().first, hierarchy.front().second);
    ConvergentInterpolationSolution result;
    Grid& solved = result.grid;
    solved = std::move(coarsest);
    // Для правила Snap нужны только шаги самого грубого уровня.
    const GridGeometry coarsestGeometry = gridGeometry(solved);

    for (std::size_t levelIndex = 0; levelIndex < hierarchy.size(); ++levelIndex) {
        const auto [nx, ny] = hierarchy[levelIndex];
        // Уточнение: первый prior уже имеет нужный грубый
        // размер; далее решение предыдущего уровня билинейно переносится на
        // более частую сетку.
        Grid prior = levelIndex == 0 ? std::move(solved) : resample(solved, nx, ny);
        const bool finalLevel = levelIndex + 1 == hierarchy.size();
        const std::size_t snapNodes = snapNodeCount(
            prior.geometry, coarsestGeometry, finalLevel, options);

        // Привязка Snap: контрольные значения проецируются по Тейлору
        // в ближайшие узлы и превращаются в диагональные мягкие
        // штрафы текущего уровня.
        std::vector<double> snapDiagonal(prior.values.size(), 0.0);
        std::vector<double> snapRhs(prior.values.size(), 0.0);
        addSnapConstraints(prior, controls, snapNodes, options, snapDiagonal, snapRhs);

        // На финальном уровне добавляется мягкая билинейная привязка C_p*u=d_p.
        // Она уменьшает величину последующей точной поправки, но не заменяет её.
        if (finalLevel) {
            result.constraints = makePointConstraints(prior.geometry, controls);
        }

        SolverResult level = solveLevel(prior.geometry, prior.values, snapDiagonal, snapRhs,
                                        result.constraints, options);
        if (!level.converged && options.throwOnNonConvergence) {
            // В сообщении указывается размер уровня, на котором не сошёлся PCG.
            // Он может отличаться от размера итоговой поверхности.
            throw std::runtime_error("Бигармонический PCG не сошёлся на уровне сетки "
                + std::to_string(nx) + "x" + std::to_string(ny) + ".");
        }
        solved = std::move(prior);
        solved.values = std::move(level.values);
    }
    return result;
}

void restoreFaultNodeValues(const Grid& input, Grid& solved)
{
    if (!input.geometry.faults) {
        return;
    }

    // Узлы на разломе не представляют ни одну сторону и сохраняют входные значения.
    for (std::size_t iy = 0; iy < solved.geometry.ny; ++iy) {
        for (std::size_t ix = 0; ix < solved.geometry.nx; ++ix) {
            if (input.geometry.faults->onFault(
                    solved.geometry.x(ix), solved.geometry.y(iy))) {
                solved.values[solved.geometry.index(ix, iy)] =
                    input.values[input.geometry.index(ix, iy)];
            }
        }
    }
}

void verifyReturnedControls(
    const Grid& solved,
    const std::vector<CanonicalPoint>& controls,
    const std::vector<PointConstraint>& outputConstraints,
    const ConvergentInterpolationOptions& options)
{
    // Внутренние и возвращаемые значения имеют один тип double.
    // В мягком режиме точная привязка не требуется.
    if (!options.enforceExactControls) {
        return;
    }

    const double controlError = maxControlError(solved, controls);
    const double acceptedControlError = effectiveControlTolerance(
        outputConstraints, options);
    if (!(controlError <= acceptedControlError)) {
        throw std::runtime_error(
            "Результат не удовлетворяет допуску точной привязки контрольных точек.");
    }
}

std::vector<double> prepareConvergentInterpolationOutput(
    ConvergentInterpolationSolution& solution,
    const Grid& input,
    const std::vector<CanonicalPoint>& controls,
    const ConvergentInterpolationOptions& options)
{
    if (options.enforceExactControls) {
        projectOntoPointConstraints(solution.grid, solution.constraints, options);
    }
    restoreFaultNodeValues(input, solution.grid);
    // Все значения проверяются до передачи буфера в Surface.
    for (double value : solution.grid.values) {
        if (!finite(value)) {
            throw std::runtime_error("Конвергентная интерполяция дала неконечный результат.");
        }
    }
    verifyReturnedControls(solution.grid, controls, solution.constraints, options);
    return std::move(solution.grid.values);
}

} // namespace

// ============================================================================
// 9. Публичный интерфейс
// ============================================================================

// Оркестратор хранит порядок стадий и время жизни геометрии разломов.
// Исключение на любой стадии до записи результата оставляет Surface неизменной.
void convergentInterpolation(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentInterpolationOptions& options)
{
    convergentInterpolation(surface, points, std::vector<Fault>{}, options);
}

void convergentInterpolation(
    Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentInterpolationOptions& options)
{
    const auto controls = validateInput(surface, points, options);
    const FaultGeometry barriers(surface, faults);
    for (const auto& point : controls) {
        if (barriers.onFault(point.x, point.y)) {
            throw std::invalid_argument(
                "Контрольная точка находится на разломе; задайте точку с одной из его сторон.");
        }
    }

    if (controls.empty()) {
        return;
    }

    // barriers живет до завершения всех стадий: Grid хранит только указатель.
    Grid input = surfaceAsGrid(surface);
    input.geometry.faults = barriers.empty() ? nullptr : &barriers;
    auto solution = solveConvergentInterpolationHierarchy(
        surface, input, controls, options);
    auto output = prepareConvergentInterpolationOutput(
        solution, input, controls, options);

    // Единственная запись в Surface: все проверки уже завершены.
    // При исключении до этой строки поверхность остается неизменной.
    surface.grid = std::move(output);
}

Surface convergentInterpolatedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentInterpolationOptions& options)
{
    // Немутирующая перегрузка реализована через локальную копию и основной
    // оркестратор, поэтому численный путь обеих публичных функций идентичен.
    return convergentInterpolatedSurface(surface, points, {}, options);
}

Surface convergentInterpolatedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentInterpolationOptions& options)
{
    Surface result = surface;
    convergentInterpolation(result, points, faults, options);
    return result;
}

double sampleSurface(const Surface& surface, double x, double y,
                    const std::vector<Fault>& faults)
{
    // Для одной выборки не копируем и не сканируем весь grid: после проверки
    // геометрии читаются только значения опорных узлов (до четырёх).
    if (surface.nx < 2 || surface.ny < 2
        || surface.grid.size() != checkedNodeCount(surface.nx, surface.ny)) {
        throw std::invalid_argument("Недопустимые размеры surface для выборки значений.");
    }
    GridGeometry grid = gridGeometry(surface, surface.nx, surface.ny);
    if (!finite(grid.minx) || !finite(grid.maxx) || !finite(grid.miny) || !finite(grid.maxy)
        || !(grid.maxx > grid.minx) || !(grid.maxy > grid.miny)
        || !finite(grid.dx()) || !finite(grid.dy()) || !(grid.dx() > 0) || !(grid.dy() > 0)
        || !finite(x) || !finite(y)
        || x < grid.minx || x > grid.maxx || y < grid.miny || y > grid.maxy) {
        throw std::invalid_argument("Недопустимые координаты выборки или границы surface.");
    }
    const FaultGeometry barriers(surface, faults);
    grid.faults = barriers.empty() ? nullptr : &barriers;
    PointConstraint row;
    if (!samplingConstraint(grid, x, y, row)) {
        throw std::invalid_argument(
            "Невозможно получить значение на разломе или в блоке, не представленном узлами сетки; "
            "увеличьте nx и ny.");
    }
    double value = 0.0;
    for (std::size_t k = 0; k < row.count; ++k) {
        const double sample = surface.grid[row.terms[k].index];
        if (!finite(sample)) {
            throw std::invalid_argument("Опорные узлы выборки содержат неконечное значение.");
        }
        value += row.terms[k].coefficient * sample;
    }
    if (!finite(value)) {
        throw std::runtime_error("Выборка дала неконечное значение.");
    }
    return value;
}

} // namespace ConvergentInterpolation
