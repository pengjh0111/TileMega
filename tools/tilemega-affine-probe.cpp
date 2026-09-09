// SPDX-License-Identifier: BSD-3-Clause
// T3: offline experiment only; no production scheduling code is changed.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ReferenceModels.h>
#include <tilemega/Solver/ListScheduler.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/point.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace tilemega::analysis;
namespace {
struct Task {
  int op;
  std::vector<long> coordinates, time;
};
using Key = std::pair<int, std::vector<long>>;
struct Enumerate {
  int op, inputs, outputs;
  std::vector<Task>* tasks;
};
isl_stat Point(isl_point* point, void* data) {
  auto& e = *static_cast<Enumerate*>(data);
  Task t{e.op, {}, {}};
  for (int i = 0; i < e.inputs + e.outputs; ++i) {
    isl_val* v = isl_point_get_coordinate_val(point, isl_dim_set, i);
    (i < e.inputs ? t.coordinates : t.time).push_back(isl_val_get_num_si(v));
    isl_val_free(v);
  }
  e.tasks->push_back(std::move(t));
  isl_point_free(point);
  return isl_stat_ok;
}
isl_stat Map(isl_map* map, void* data) {
  char const* name = isl_map_get_tuple_name(map, isl_dim_in);
  Enumerate e{std::stoi(std::string(name).substr(1)),
              static_cast<int>(isl_map_dim(map, isl_dim_in)),
              static_cast<int>(isl_map_dim(map, isl_dim_out)),
              static_cast<std::vector<Task>*>(data)};
  isl_set* flat = isl_set_flatten(isl_map_wrap(map));
  auto result = isl_set_foreach_point(flat, Point, &e);
  isl_set_free(flat);
  return result;
}
isl_bool Band(isl_schedule_node* node, void* data) {
  if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
    ++*static_cast<int*>(data);
    std::cout << "BAND depth=" << isl_schedule_node_get_schedule_depth(node)
              << " members=" << isl_schedule_node_band_n_member(node)
              << " permutable=" << isl_schedule_node_band_get_permutable(node) << '\n';
  }
  return isl_bool_true;
}
void Print(char const* label, char* text) {
  if (!text) throw std::runtime_error(std::string("isl null result: ") + label);
  std::cout << label << ' ' << text << '\n';
  std::free(text);
}
struct BandPlacement {
  std::map<Key, std::pair<int, long>> coordinate;
  std::map<int, std::pair<long, long>> bounds;
  int next = 0;
};
isl_bool CollectBand(isl_schedule_node* node, void* data) {
  if (isl_schedule_node_get_type(node) != isl_schedule_node_band ||
      !isl_schedule_node_band_get_permutable(node)) return isl_bool_true;
  int axis = -1;
  for (int i = 0; i < isl_schedule_node_band_n_member(node); ++i)
    if (isl_schedule_node_band_member_get_coincident(node, i) == isl_bool_true) {
      axis = i; break;
    }
  if (axis < 0) return isl_bool_true;
  auto& placement = *static_cast<BandPlacement*>(data);
  int const band = placement.next++;
  auto* partial = isl_union_map_intersect_domain(
      isl_schedule_node_band_get_partial_schedule_union_map(node),
      isl_schedule_node_get_domain(node));
  std::vector<Task> points;
  if (isl_union_map_foreach_map(partial, Map, &points) != isl_stat_ok) {
    isl_union_map_free(partial);
    return isl_bool_error;
  }
  isl_union_map_free(partial);
  for (auto const& point : points) {
    Key const key{point.op, point.coordinates};
    // Top-down visitation keeps the outermost eligible band for each task.
    if (placement.coordinate.count(key)) continue;
    long value = point.time.at(axis);
    placement.coordinate[key] = {band, value};
    auto [it, inserted] = placement.bounds.emplace(band, std::make_pair(value, value));
    if (!inserted) {
      it->second.first = std::min(it->second.first, value);
      it->second.second = std::max(it->second.second, value);
    }
  }
  return isl_bool_true;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4) throw std::runtime_error("usage: tilemega-affine-probe SEQ WORKERS RESIDENT_LIMIT");
  int const seq = std::stoi(argv[1]), workers = std::stoi(argv[2]);
  int const resident = std::stoi(argv[3]);
  if (seq <= 0 || workers <= 0 || resident <= 0) throw std::runtime_error("positive arguments required");
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [k, v] : DecoderShape::Table27G().values) known.Bind(k, v);
  known.Bind("H", 512); known.Bind("I", 1024); known.Bind("n_h", 4);
  known.Bind("n_kv", 2); known.Bind("G", 2); known.Bind("d", 128);
  known.Bind("S", seq); known.Bind("past", 3); known.Bind("L_s", seq + 3);
  auto graph = LlamaDecoderLayer(DecoderShape{});
  auto edges = CouplingDerivation().Derive(graph, known);
  isl_ctx* ctx = SharedIslContext().raw();
  isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
  std::map<std::string, int> ids;
  isl_union_set* domain = isl_union_set_read_from_str(ctx, "{ }");
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    auto const& n = graph.nodes[i]; ids[n.name] = i;
    std::vector<long> extents;
    for (std::size_t axis = 0; axis < n.output.axes.size(); ++axis)
      if (n.IsTiled(axis)) extents.push_back(n.CoordinateExtent(axis).Eval(known, {}));
    std::ostringstream text;
    text << "{ O" << i << '[';
    for (std::size_t j = 0; j < extents.size(); ++j) text << (j ? "," : "") << 'x' << j;
    text << ']';
    for (std::size_t j = 0; j < extents.size(); ++j)
      text << (j ? " and " : " : ") << "0 <= x" << j << " < " << extents[j];
    text << " }";
    domain = isl_union_set_add_set(domain, isl_set_read_from_str(ctx, text.str().c_str()));
  }
  isl_union_map* validity = isl_union_map_read_from_str(ctx, "{ }");
  std::vector<std::pair<Key, Key>> dependencies;
  std::vector<std::vector<int>> successors(graph.nodes.size());
  for (auto const& edge : edges) {
    int const src = ids.at(edge.src.name), dst = ids.at(edge.dst.name);
    successors[src].push_back(dst);
    auto relation = edge.C.BindParams(known);
    isl_map* map = isl_map_reverse(isl_map_read_from_str(ctx, relation.ToString().c_str()));
    map = isl_map_set_tuple_name(map, isl_dim_in, ("O" + std::to_string(src)).c_str());
    map = isl_map_set_tuple_name(map, isl_dim_out, ("O" + std::to_string(dst)).c_str());
    // Coupling windows can extend past a partial boundary tile (or into the
    // already-existing KV prefix). Only points in the actual task domains
    // are scheduled. This is the same domain intersection used at launch.
    auto* producer_domain = isl_union_set_extract_set(
        domain, isl_space_domain(isl_map_get_space(map)));
    auto* consumer_domain = isl_union_set_extract_set(
        domain, isl_space_range(isl_map_get_space(map)));
    map = isl_map_intersect_domain(map, producer_domain);
    map = isl_map_intersect_range(map, consumer_domain);
    char* clipped_text = isl_map_to_str(map);
    auto const clipped = CouplingRelation::FromIslText(clipped_text);
    std::free(clipped_text);
    validity = isl_union_map_add_map(validity, map);
    for (auto const& [producer, consumer] : clipped.Points())
      dependencies.push_back({{src, producer}, {dst, consumer}});
  }
  Print("DOMAIN", isl_union_set_to_str(domain));
  Print("VALIDITY_PRODUCER_TO_CONSUMER", isl_union_map_to_str(validity));
  auto* constraints = isl_schedule_constraints_on_domain(isl_union_set_copy(domain));
  constraints = isl_schedule_constraints_set_validity(constraints, isl_union_map_copy(validity));
  constraints = isl_schedule_constraints_set_proximity(constraints, isl_union_map_copy(validity));
  isl_schedule* schedule = isl_schedule_constraints_compute_schedule(constraints);
  if (!schedule) {
    std::cerr << "ISL_ERROR " << isl_ctx_last_error_msg(ctx) << '\n';
    isl_union_map_free(validity);
    return 2;
  }
  Print("SCHEDULE", isl_schedule_to_str(schedule));
  int bands = 0;
  isl_schedule_foreach_schedule_node_top_down(schedule, Band, &bands);
  BandPlacement band_placement;
  if (isl_schedule_foreach_schedule_node_top_down(schedule, CollectBand, &band_placement) < 0)
    throw std::runtime_error("cannot enumerate outermost parallel bands");
  isl_union_map* mapping = isl_union_map_intersect_domain(
      isl_schedule_get_map(schedule), domain);
  Print("SCHEDULE_MAP", isl_union_map_to_str(mapping));
  isl_union_map* ordered = isl_union_map_lex_lt_union_map(isl_union_map_copy(mapping), isl_union_map_copy(mapping));
  isl_union_map* bad = isl_union_map_subtract(isl_union_map_copy(validity), ordered);
  if (isl_union_map_is_empty(bad) != isl_bool_true) {
    Print("ILLEGAL_EDGES", isl_union_map_to_str(bad));
    throw std::runtime_error("schedule violates validity");
  }
  isl_union_map_free(bad);
  std::vector<Task> tasks;
  if (isl_union_map_foreach_map(mapping, Map, &tasks) != isl_stat_ok)
    throw std::runtime_error("cannot enumerate schedule");
  std::size_t dimensions = 0;
  for (auto const& task : tasks) dimensions = std::max(dimensions, task.time.size());
  std::cout << "DIMENSIONS " << dimensions << '\n';
  std::map<Key, int> task_ids;
  for (std::size_t i = 0; i < tasks.size(); ++i) task_ids[{tasks[i].op, tasks[i].coordinates}] = i;
  long expected = 0;
  for (auto const& node : graph.nodes) expected += node.Count().Eval(known, {});
  if (task_ids.size() != tasks.size() || tasks.size() != static_cast<std::size_t>(expected))
    throw std::runtime_error("schedule lost or duplicated tasks");
  auto stage_order = tilemega::solver::ListScheduler().Schedule(successors);
  std::vector<int> rank(stage_order.size());
  for (std::size_t i = 0; i < stage_order.size(); ++i) rank[stage_order[i]] = i;
  for (std::string const mode : {"stage_major", "affine", "band_tiling", "wavefront"}) {
    bool const affine = mode != "stage_major";
    std::vector<int> order(tasks.size()); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      auto const& x = tasks[a]; auto const& y = tasks[b];
      if (affine && x.time != y.time) return x.time < y.time;
      if (rank[x.op] != rank[y.op]) return rank[x.op] < rank[y.op];
      return x.coordinates < y.coordinates;
    });
    std::vector<int> worker(tasks.size()), slot(tasks.size()), lengths(workers, 0);
    std::vector<std::vector<int>> task_successors(tasks.size());
    std::vector<int> indegree(tasks.size(), 0), last(workers, -1);
    auto add_dependency = [&](int from, int to) {
      task_successors[from].push_back(to); ++indegree[to];
    };
    int previous = -1, logical = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      int const id = order[i];
      if (tasks[id].op != previous) { previous = tasks[id].op; logical = 0; }
      int w = affine ? i % workers : logical++ % workers;
      if (mode == "band_tiling") {
        auto const found = band_placement.coordinate.find({tasks[id].op, tasks[id].coordinates});
        if (found == band_placement.coordinate.end())
          throw std::runtime_error("band mapping has a task without a parallel band");
        auto const [band, value] = found->second;
        auto const [low, high] = band_placement.bounds.at(band);
        long const tile_width = (high - low + 1 + workers - 1) / workers;
        w = static_cast<int>((value - low) / tile_width);
      } else if (mode == "wavefront") {
        if (tasks[id].time.size() < 2)
          throw std::runtime_error("wavefront needs two schedule dimensions");
        // time[0] is the wave; lexicographic ordering preserves its order.
        // Keep equal second coordinates on one worker within every wave.
        w = static_cast<int>((tasks[id].time[1] % workers + workers) % workers);
      }
      worker[id] = w; slot[id] = lengths[w]++;
      if (last[w] >= 0) add_dependency(last[w], id);
      last[w] = id;
    }
    std::vector<int> spans;
    int forward_worker_span = 0;
    std::size_t same_worker = 0;
    for (auto const& [producer, consumer] : dependencies) {
      int a = task_ids.at(producer), b = task_ids.at(consumer);
      add_dependency(a, b);
      spans.push_back(slot[b] - slot[a]);
      same_worker += worker[a] == worker[b];
      forward_worker_span = std::max(forward_worker_span, worker[a] - worker[b]);
    }
    std::vector<int> ready;
    for (std::size_t i = 0; i < tasks.size(); ++i) if (!indegree[i]) ready.push_back(i);
    for (std::size_t i = 0; i < ready.size(); ++i)
      for (int next : task_successors[ready[i]]) if (!--indegree[next]) ready.push_back(next);
    if (ready.size() != tasks.size()) throw std::runtime_error("worker queue introduces a cycle");
    std::sort(spans.begin(), spans.end());
    if (spans.empty()) throw std::runtime_error("no dependency spans");
    std::cout << "SPAN mode=" << mode
              << " seq=" << seq << " workers=" << workers << " resident_limit=" << resident
              << " tasks=" << tasks.size() << " edges=" << spans.size()
              << " same_worker_edges=" << same_worker
              << " cross_worker_fraction=" << double(spans.size()-same_worker)/spans.size()
              << " same_worker_fraction=" << double(same_worker)/spans.size()
              << " used_workers=" << std::count_if(lengths.begin(), lengths.end(), [](int n) { return n > 0; })
              << " max_queue=" << *std::max_element(lengths.begin(), lengths.end())
              << " min=" << spans.front() << " p50=" << spans[spans.size()/2]
              << " p95=" << spans[spans.size()*95/100] << " max=" << spans.back()
              << " forward_worker_span=" << forward_worker_span
              << " slot_span_below_limit=" << (spans.back() < resident)
              << " resident_grid_safe=" << (workers <= resident)
              << " queue_cycle=0"
              << " overresident_proven=0\n";
    std::map<int, std::size_t> distribution;
    for (int span : spans) ++distribution[span];
    for (auto const& [span, count] : distribution)
      std::cout << "HIST mode=" << (affine ? "affine" : "stage_major")
                << " slot_span=" << span << " edges=" << count << '\n';
  }
  std::cout << "RESULT legal=1 bands=" << bands << " bounded_at_fixed_parameters=1\n";
  isl_union_map_free(mapping); isl_union_map_free(validity); isl_schedule_free(schedule);
  return 0;
} catch (std::exception const& e) {
  std::cerr << "affine-probe: " << e.what() << '\n'; return 1;
}
