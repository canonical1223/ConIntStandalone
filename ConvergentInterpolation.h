#pragma once

#include <cstddef>
#include <vector>

namespace ConvergentInterpolation {

// Входные данные
// -------------

// min/max — координаты крайних узлов; nx,ny >= 2, grid.size() == nx*ny.
// grid[iy*nx+ix]: слева направо, снизу вверх; grid[0] — (minx,miny).
// Входные значения задают исходную форму поверхности и мягкую регуляризацию.
struct Surface {
    std::vector<double> grid;
    double minx{};
    double maxx{};
    double miny{};
    double maxy{};
    std::size_t nx{};
    std::size_t ny{};
};

// Конечные координаты внутри замкнутых границ Surface. Совпадающие x/y
// объединяются во взвешенное среднее; weight == 0 игнорируется, weight < 0 запрещен.
struct Point {
    double x{};
    double y{};
    double value{};
    double weight{1.0};
};

// Последовательная XY-ломаная; value/weight не используются. Замыкание явно
// повторяет первую вершину. После удаления соседних повторов нужны две вершины.
using Fault = std::vector<Point>;

// Настройки алгоритма без значений по умолчанию.
// Создавайте options{} и явно задавайте все поля до передачи в алгоритм.
struct ConvergentInterpolationOptions {
    // Иерархия сеток
    // Число Snap-узлов масштабируется с шагом; на финале равно 1.
    std::size_t initialSnapNodes;
    // Желаемый предел интервалов по каждой оси; maxLevels ограничивает иерархию.
    std::size_t coarsestIntervals;
    std::size_t maxLevels;

    // Сглаживание и мягкая привязка
    // Энергия: smoothness*||B*u||² + priorWeight*||u-prior||² + Snap/Taylor
    // и билинейные штрафы точек на финале. B — вторые разности со свободным краем.
    // priorWeight > 0 обеспечивает SPD; остальные силы могут быть нулевыми.
    double smoothness;
    double priorWeight;
    double snapStrength;
    double finalPointStrength;

    // Точная финальная привязка
    // Отдельная проекция C*u=d минимизирует узловую L2-поправку через Gram-PCG.
    bool enforceExactControls;
    std::size_t maxControlProjectionIterations;
    // Порог: max(controlTolerance, 64*epsilon(double)*max(1,max|point.value|)).
    double controlTolerance;

    // Перенос значений точек в узлы: Snap/Тейлор
    // Сигма Snap в ячейках текущего уровня.
    double gaussianSigma;
    // 0: значение; 1: +градиент; 2: +Гессиан prior.
    int taylorOrder;
    // Делить объединенные веса на максимум, сохраняя их отношения.
    bool normalizePointWeights;

    // Основной итерационный решатель
    // PCG с предобуславливателем Якоби: ||r|| <= absTol + relTol*max(||b||,||A*x0||).
    // Лимит на каждый уровень; при throwOnNonConvergence=false принимается последняя итерация.
    std::size_t maxSolverIterations;
    double relativeTolerance;
    double absoluteTolerance;
    bool throwOnNonConvergence;
};

// Расчет поверхности и выборка значений
// ------------------------------------

// convergentInterpolation меняет поверхность; convergentInterpolatedSurface
// возвращает копию. Оба варианта используют один и тот же путь расчета.
// Проверка -> канонизация -> [уточнение -> Snap/Тейлор -> сглаживание PCG] ->
// необязательная точная привязка -> проверка результата -> запись в surface.
// При исключении surface неизменна; без активных точек сохраняются все её биты.
void convergentInterpolation(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentInterpolationOptions& options);

// Разломы блокируют локальные связи; влияние может обойти свободный конец.
// Точки на разломе запрещены, узлы на нем сохраняют входные значения.
// Непредставимые грубые уровни пропускаются; на конечной сетке это ошибка.
// options передается явно во всех перегрузках.
void convergentInterpolation(
    Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentInterpolationOptions& options);

// Немутирующие перегрузки возвращают измененную копию поверхности.
Surface convergentInterpolatedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentInterpolationOptions& options);

Surface convergentInterpolatedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentInterpolationOptions& options);

// Билинейная выборка; с разломами — перенормировка взаимно видимых углов либо
// ближайший видимый узел в радиусе двух ячеек. Выборка на разломе — ошибка.
// Тот же оператор задает привязку точек. Surface не хранит разломы: передавайте
// те же faults при выборке и разрывайте связи через них при отрисовке.
double sampleSurface(
    const Surface& surface, double x, double y,
    const std::vector<Fault>& faults = {});

} // namespace ConvergentInterpolation
