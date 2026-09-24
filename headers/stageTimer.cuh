#ifndef STAGE_TIMER_CUH
#define STAGE_TIMER_CUH

#include <chrono>
#include <iosfwd>
#include <string>

/**
 * @brief Optional stage instrumentation (wall-clock stage timers, NVTX
 *        ranges and counters).
 *
 * @details
 * Disabled by default. When disabled, ScopedStage and recordCounter() do
 * nothing, so the instrumented code paths run exactly as without them.
 * When enabled (e.g. `bin/mosp --timing stages.csv`), each ScopedStage
 * records its wall time and opens an NVTX range (visible in nsys). With
 * syncDevice = true the device is synchronized when the stage opens and
 * closes, so asynchronous GPU work is charged to the right stage; this is
 * the only behavioural difference and only happens when enabled.
 */
void setInstrumentation(bool enabled);
bool instrumentationEnabled();

/**
 * @brief Prefix prepended to every stage and counter name recorded from now
 *        on (e.g. "obj1/"); lets a driver label the stages of a call.
 */
void setStagePrefix(const std::string &prefix);
const std::string &stagePrefix();

/** @brief Append a stage record (milliseconds). */
void recordStage(const std::string &name, double milliseconds);

/** @brief Add @p value to a named counter. */
void recordCounter(const std::string &name, double value);

/** @brief Sum of all stage records whose name ends with @p suffix. */
double totalStageTime(const std::string &suffix);

/** @brief Drop all records and counters. */
void clearInstrumentation();

/** @brief Print "STAGE name ms" and "COUNTER name value" lines. */
void printInstrumentation(std::ostream &out);

/** @brief Write the records as CSV: kind,name,value. */
bool writeInstrumentationCsv(const std::string &path);

/** @brief Synchronize the device (no-op without CUDA or when disabled). */
void instrumentationSync();

/** @brief RAII stage timer; see file comment. */
class ScopedStage {
public:
  explicit ScopedStage(const std::string &name, bool syncDevice = false);
  ~ScopedStage() { stop(); }
  ScopedStage(const ScopedStage &) = delete;
  ScopedStage &operator=(const ScopedStage &) = delete;
  void stop();

private:
  std::string name_;
  bool active_;
  bool sync_;
  std::chrono::steady_clock::time_point start_;
};

#endif // STAGE_TIMER_CUH
