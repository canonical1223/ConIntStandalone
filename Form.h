#pragma once

#include <QWidget>
#include <cstddef>
#include <memory>

namespace Ui { class Form; }

// Только интерфейс, отдельные значения, обработчики и сигналы изменений.
// Form не знает о типах и правилах вычислительного модуля.
class Form : public QWidget {
    Q_OBJECT

public:
    explicit Form(QWidget* parent = nullptr);
    ~Form() override;

    std::size_t getInitialSnapNodes() const noexcept { return initialSnapNodes_; }
    std::size_t getCoarsestIntervals() const noexcept { return coarsestIntervals_; }
    std::size_t getMaxLevels() const noexcept { return maxLevels_; }
    qreal getSmoothness() const noexcept { return smoothness_; }
    qreal getPriorWeight() const noexcept { return priorWeight_; }
    qreal getSnapStrength() const noexcept { return snapStrength_; }
    qreal getFinalPointStrength() const noexcept { return finalPointStrength_; }
    bool getEnforceExactControls() const noexcept { return enforceExactControls_; }
    std::size_t getMaxControlProjectionIterations() const noexcept { return maxControlProjectionIterations_; }
    qreal getControlTolerance() const noexcept { return controlTolerance_; }
    qreal getGaussianSigma() const noexcept { return gaussianSigma_; }
    int getTaylorOrder() const noexcept { return taylorOrder_; }
    bool getNormalizePointWeights() const noexcept { return normalizePointWeights_; }
    std::size_t getMaxSolverIterations() const noexcept { return maxSolverIterations_; }
    qreal getRelativeTolerance() const noexcept { return relativeTolerance_; }
    qreal getAbsoluteTolerance() const noexcept { return absoluteTolerance_; }
    bool getThrowOnNonConvergence() const noexcept { return throwOnNonConvergence_; }

signals:
    // Сигнал отправляется после записи нового значения, только при его изменении.
    void initialSnapNodesChanged(std::size_t value);
    void coarsestIntervalsChanged(std::size_t value);
    void maxLevelsChanged(std::size_t value);
    void smoothnessChanged(qreal value);
    void priorWeightChanged(qreal value);
    void snapStrengthChanged(qreal value);
    void finalPointStrengthChanged(qreal value);
    void enforceExactControlsChanged(bool value);
    void maxControlProjectionIterationsChanged(std::size_t value);
    void controlToleranceChanged(qreal value);
    void gaussianSigmaChanged(qreal value);
    void taylorOrderChanged(int value);
    void normalizePointWeightsChanged(bool value);
    void maxSolverIterationsChanged(std::size_t value);
    void relativeToleranceChanged(qreal value);
    void absoluteToleranceChanged(qreal value);
    void throwOnNonConvergenceChanged(bool value);

private slots:
    // Подключаются автоматически через setupUi()/connectSlotsByName().
    void on_initialSnapNodes_textChanged(const QString& text);
    void on_coarsestIntervals_textChanged(const QString& text);
    void on_maxLevels_textChanged(const QString& text);
    void on_smoothness_textChanged(const QString& text);
    void on_priorWeight_textChanged(const QString& text);
    void on_snapStrength_textChanged(const QString& text);
    void on_finalPointStrength_textChanged(const QString& text);
    void on_enforceExactControls_toggled(bool value);
    void on_maxControlProjectionIterations_textChanged(const QString& text);
    void on_controlTolerance_textChanged(const QString& text);
    void on_gaussianSigma_textChanged(const QString& text);
    void on_taylorOrder_currentIndexChanged(int value);
    void on_normalizePointWeights_toggled(bool value);
    void on_maxSolverIterations_textChanged(const QString& text);
    void on_relativeTolerance_textChanged(const QString& text);
    void on_absoluteTolerance_textChanged(const QString& text);
    void on_throwOnNonConvergence_toggled(bool value);

private:
    std::unique_ptr<Ui::Form> ui_;

    std::size_t initialSnapNodes_{16};
    std::size_t coarsestIntervals_{8};
    std::size_t maxLevels_{8};
    qreal smoothness_{qreal(1)};
    qreal priorWeight_{qreal(1e-3)};
    qreal snapStrength_{qreal(100)};
    qreal finalPointStrength_{qreal(1000)};
    bool enforceExactControls_{true};
    std::size_t maxControlProjectionIterations_{2000};
    qreal controlTolerance_{qreal(1e-10)};
    qreal gaussianSigma_{qreal(1)};
    int taylorOrder_{2};
    bool normalizePointWeights_{true};
    std::size_t maxSolverIterations_{1500};
    qreal relativeTolerance_{qreal(1e-9)};
    qreal absoluteTolerance_{qreal(0)};
    bool throwOnNonConvergence_{true};
};
