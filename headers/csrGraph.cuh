#ifndef CSR_GRAPH_CUH
#define CSR_GRAPH_CUH

#include <functional>
#include <string>
#include <vector>

/**
 * @brief Multi-objective directed graph in CSR form (one row per source).
 *
 * @details
 * Row u lists the out-edges of u. Each edge carries numberOfObjectives
 * integer weights stored edge-major, i.e. objective k of edge e is
 * weights[e * numberOfObjectives + k] (the layout of the Values text file).
 * A reverse graph built by transposeCsrGraph() uses the same type with
 * rows indexed by destination.
 */
struct CsrGraph {
  int numberOfNodes = 0;
  int numberOfObjectives = 0;
  std::vector<int> rowPtr;  ///< numberOfNodes + 1 offsets
  std::vector<int> colInd;  ///< neighbour of each edge
  std::vector<int> weights; ///< numberOfObjectives weights per edge

  int numberOfEdges() const { return rowPtr.empty() ? 0 : rowPtr.back(); }
  int weight(int edge, int objective) const {
    return weights[static_cast<size_t>(edge) * numberOfObjectives + objective];
  }
};

/**
 * @brief A batch of edge changes (the contents of insert.txt / delete.txt).
 *
 * @details
 * Insertions carry numberOfObjectives weights each (insertWeights is
 * insertion-major). weightIncreaseMask is filled by applyChangeBatch():
 * bit k of entry i is set when insertion i overwrote an edge that already
 * existed in the original graph and its final objective-k weight is larger
 * than the original one (a "weight increase", which invalidates the tree
 * edge like a deletion does).
 */
struct ChangeBatch {
  int numberOfObjectives = 0;
  std::vector<int> insertFrom, insertTo;
  std::vector<int> insertWeights;
  std::vector<int> deleteFrom, deleteTo;
  std::vector<unsigned int> weightIncreaseMask;

  int numberOfInserts() const { return static_cast<int>(insertFrom.size()); }
  int numberOfDeletes() const { return static_cast<int>(deleteFrom.size()); }
};

/**
 * @brief Read <prefix>RowPtr.txt, <prefix>ColInd.txt and <prefix>Values.txt.
 *
 * @details
 * Rejects (with a message naming the file and line) row pointers that do
 * not start at 0 or decrease, column indices outside [0, n), ids or offsets
 * above INT_MAX, and weights outside [1, INT_MAX].
 */
bool readCsrGraph(const std::string &prefix, CsrGraph &graph);

/** @brief Write a graph in the same three-file text format. */
bool writeCsrGraph(const std::string &prefix, const CsrGraph &graph);

/**
 * @brief Save a graph as one binary file, a cache of the text graph at
 *        @p sourcePrefix.
 *
 * @details
 * The file records the identity of its source: the canonical path of
 * @p sourcePrefix and the size and modification time (nanoseconds) of the
 * three text files, taken when the cache is written.
 */
bool saveCsrGraphBinary(const std::string &path, const CsrGraph &graph,
                        const std::string &sourcePrefix);

/**
 * @brief Load a graph written by saveCsrGraphBinary() for the text graph
 *        at @p sourcePrefix.
 *
 * @details
 * Fails (without a message) if the file is missing, truncated or of an
 * older format, if its recorded identity differs from the current text
 * files at @p sourcePrefix (another graph, or files changed or copied
 * since), or if the arrays are not a valid CSR graph (row pointers from 0,
 * monotone, ending at m; column indices in [0, n); weights in
 * [1, INT_MAX]).
 */
bool loadCsrGraphBinary(const std::string &path, CsrGraph &graph,
                        const std::string &sourcePrefix);

/**
 * @brief Load a text CSR graph, optionally through a binary cache.
 *
 * @details
 * With a non-empty @p cachePath the binary file is used when
 * loadCsrGraphBinary() accepts it for @p prefix; otherwise the text is
 * parsed and the cache is (re)written, with a notice if a cache file was
 * there but did not match.
 */
bool loadCsrGraph(const std::string &prefix, CsrGraph &graph,
                  const std::string &cachePath = "");

/**
 * @brief Read insert.txt ("u v w1 .. wK") and delete.txt ("u v").
 *
 * @details
 * Endpoints must be in [0, numberOfNodes) and inserted weights in
 * [1, INT_MAX]; errors name the file and line.
 *
 * @param numberOfObjectives Weights expected per insertion.
 * @param numberOfNodes      Vertex count used to validate endpoints.
 */
bool readChangeBatch(const std::string &insertPath,
                     const std::string &deletePath, int numberOfObjectives,
                     int numberOfNodes, ChangeBatch &batch);

/**
 * @brief Apply a batch to a graph, producing the updated graph.
 *
 * @details
 * Same semantics as updateGraphCSR() and the adjacency-list edits of the
 * original SOSP update: deletions first (each removes the first remaining
 * (u,v) edge), then insertions in file order (an existing (u,v) edge gets
 * its weights overwritten, otherwise the edge is appended; the last
 * insertion of a pair wins). Surviving edges keep their order and new edges
 * are appended to their row. Also fills batch.weightIncreaseMask.
 */
bool applyChangeBatch(const CsrGraph &original, ChangeBatch &batch,
                      CsrGraph &updated);

/** @brief Build the reverse graph: row v lists u for every edge u -> v. */
void transposeCsrGraph(const CsrGraph &graph, CsrGraph &reverse);

/**
 * @brief Read a distance file ("v d" or "v INF" per line).
 *
 * @details
 * Every vertex must be listed exactly once and distances must not be
 * negative (an incomplete file would silently leave vertices at INF).
 */
bool readDistances(const std::string &path, int numberOfNodes,
                   std::vector<long long> &distances);

/**
 * @brief Read a parent file ("v p" per line, p = -1 for none).
 *
 * @details
 * Every vertex must be listed exactly once, with p in [-1, numberOfNodes).
 */
bool readParents(const std::string &path, int numberOfNodes,
                 std::vector<int> &parent);

/** @brief Write distances; values >= INF/2 are written as "INF". */
bool writeDistances(const std::string &path,
                    const std::vector<long long> &distances);

/** @brief Write a parent array. */
bool writeParents(const std::string &path, const std::vector<int> &parent);

/**
 * @brief Run independent jobs (e.g. file reads/writes) concurrently.
 *
 * @details
 * With OpenMP the jobs run on the OpenMP worker threads, which respects
 * thread pinning (threads started with std::async would inherit the
 * single-core affinity of a pinned master thread); without OpenMP each
 * job runs in its own std::async thread.
 *
 * @return true if every job returned true.
 */
bool runConcurrently(const std::vector<std::function<bool()>> &jobs);

/**
 * @brief Parse a whole decimal integer (command-line values).
 *
 * @return false if @p text is empty, has trailing characters, or the value
 *         is outside [@p minimum, @p maximum].
 */
bool parseInteger(const std::string &text, long long minimum,
                  long long maximum, long long &value);

/** @brief Distance sentinel used throughout (as in the original code). */
constexpr long long DISTANCE_INF = 0x7fffffffffffffffLL / 4;

#endif // CSR_GRAPH_CUH
