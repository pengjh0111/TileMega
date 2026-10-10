// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <algorithm>
#include <cassert>
#include <set>
#include <sstream>

namespace tilemega::tests::window_task_access_test {
namespace {
using namespace analysis;
using Point = std::pair<std::vector<long>, std::vector<long>>;
using Points = std::set<Point>;
ClosedForm F(long value) { return ClosedForm::Constant(value); }
TensorAxis Axis(char const* name, long extent, long origin = 0) {
  TensorAxis out; out.name = name; out.extent = F(extent); out.origin = F(origin); return out;
}
IterationDim Dim(char const* name, long extent) {
  IterationDim out; out.name = name; out.extent = F(extent); return out;
}
IndexResult I(char const* name, long scale = 1, long group = 1) {
  return IndexResult::Dim(name, F(scale), F(group));
}
IndexResult Sum(std::initializer_list<IndexResult> values, long offset = 0) {
  std::vector<IndexResult::Term> terms;
  for (auto const& value : values) terms.insert(terms.end(), value.terms.begin(), value.terms.end());
  return IndexResult::Affine(std::move(terms), F(offset));
}
std::vector<long> Coordinate(OperatorNode const& task, std::vector<long> const& full) {
  assert(full.size() == task.output.axes.size());
  std::vector<long> result;
  for (unsigned axis = 0; axis < full.size(); ++axis)
    if (task.IsTiled(axis)) result.push_back(full[axis]);
  return result;
}
Points PointsOf(CouplingRelation const& map) {
  auto points = map.Points(); return {points.begin(), points.end()};
}
CouplingRelation Enumerated(Points const& points, unsigned domain, unsigned range) {
  auto tuple = [](std::vector<long> const& point) {
    std::string out = "[";
    for (auto value : point) { if (out.size() > 1) out += ","; out += std::to_string(value); }
    return out + "]";
  };
  std::string text = "{ ";
  for (auto const& [left, right] : points) text += tuple(left) + " -> " + tuple(right) + "; ";
  if (points.empty()) text += tuple(std::vector<long>(domain)) + " -> " +
      tuple(std::vector<long>(range)) + " : false";
  return CouplingRelation::FromIslText(text + " }");
}
void Check(CouplingRelation const& actual, Points const& expected) {
  auto listed = Enumerated(expected, actual.DomainDimNames().size(), actual.RangeDimNames().size());
  assert(Contains(actual, listed) && Contains(listed, actual));
  assert(PointsOf(actual) == expected);
}
void Window(int n, int h, int w, int c, int o, int r, int s, int stride,
            int pad, int dilation, bool channelwise, int tm, int tn, int split = 0) {
  int p = (h + 2 * pad - dilation * (r - 1) - 1) / stride + 1;
  int q = (w + 2 * pad - dilation * (s - 1) - 1) / stride + 1;
  constexpr int producer_m = 4, producer_c = 3;
  SemanticOp producer;
  producer.domain = {Dim("m", n * h * w), Dim("c", c)};
  TensorSpace tensor{"input", {Axis("n", n), Axis("c", c), Axis("h", h), Axis("w", w)}};
  producer.result = tensor;
  producer.result_map.results = {I("m", 1, h * w), I("c"),
      Sum({I("m", 1, w), I("m", -h, h * w)}), Sum({I("m"), I("m", -w, w)})};
  OperatorNode writer;
  writer.output = {"ownership", {Axis("m", n * h * w), Axis("c", c)}};
  writer.tile = {F(producer_m), F(producer_c)};
  TaskElementPartition wp; wp.ownership.results = {I("m"), I("c")};
  auto writes = ProjectTaskElements(producer, writer, wp, tensor, producer.result_map, {});
  SemanticOp consumer;
  consumer.domain = {Dim("m", n * p * q), Dim("oc", o), Dim("r", r), Dim("s", s)};
  if (!channelwise) consumer.domain.push_back(Dim("c", c));
  for (auto& dim : consumer.domain)
    if (dim.name == "r" || dim.name == "s" || dim.name == "c")
      dim.type = IteratorType::kReduction;
  OperatorNode reader;
  reader.output = {"ownership", {Axis("m", n * p * q), Axis("oc", o)}};
  reader.tile = {F(tm), F(tn)};
  TaskElementPartition rp; rp.ownership.results = {I("m"), I("oc")};
  if (split) {
    assert(!channelwise);
    consumer.reduction.splittable = true; consumer.reduction.dim = "c";
    reader.output.axes.push_back(Axis("j", (c + split - 1) / split));
    reader.tile.push_back(F(1)); rp.reduction_chunk = F(split);
  }
  auto halo = tensor;
  halo.axes[2] = Axis("h", h + 2 * pad, -pad);
  halo.axes[3] = Axis("w", w + 2 * pad, -pad);
  IndexingMap map{{I("m", 1, p * q), channelwise ? I("oc") : I("c"),
      Sum({I("m", stride, q), I("m", -stride * p, p * q), I("r", dilation)}, -pad),
      Sum({I("m", stride), I("m", -stride * q, q), I("s", dilation)}, -pad)}};
  auto reads = ProjectTaskElements(consumer, reader, rp, halo, map, {});
  auto implicit=consumer;auto window_map=map;
  implicit.domain.erase(std::remove_if(implicit.domain.begin(),implicit.domain.end(),
      [](auto const& dim){return dim.name=="r" || dim.name=="s";}),implicit.domain.end());
  for(unsigned axis=2;axis<4;++axis) {
    auto& index=window_map.results[axis];
    index.terms.erase(std::remove_if(index.terms.begin(),index.terms.end(),
        [](auto const& term){return term.dim=="r" || term.dim=="s";}),index.terms.end());
    index.span=F(axis==2?r:s);index.window_stride=F(dilation);
  }
  implicit.operands={{"",halo,window_map}};
  implicit=DecodeSemanticOp(EncodeSemanticOp(implicit));
  auto expanded=ProjectTaskRead(implicit,reader,rp,halo,implicit.operands[0].map,{},{});
  assert(Contains(expanded,reads) && Contains(reads,expanded));
  auto exact = DeriveExactTaskCoupling(writes, reads, reader);
  auto coupling = exact.relation;
  Points expected_reads, expected_edges;
  for (int pixel = 0; pixel < n * p * q; ++pixel) for (int out = 0; out < o; ++out)
    for (int kr = 0; kr < r; ++kr) for (int ks = 0; ks < s; ++ks)
      for (int channel = channelwise ? out : 0; channel < (channelwise ? out + 1 : c); ++channel) {
        int image = pixel / (p * q), ph = pixel / q % p, pw = pixel % q;
        int ih = ph * stride + kr * dilation - pad, iw = pw * stride + ks * dilation - pad;
        std::vector<long> task{pixel / tm, out / tn};
        if (split) task.push_back(channel / split);
        task = Coordinate(reader, task);
        expected_reads.insert({task, {image, channel, ih, iw}});
        if (ih >= 0 && ih < h && iw >= 0 && iw < w)
          expected_edges.insert({task, Coordinate(writer,
              {(image * h * w + ih * w + iw) / producer_m, channel / producer_c})});
      }
  assert(PointsOf(reads) == expected_reads);
  Check(coupling, expected_edges);
  auto table = BuildDependencyTable(coupling, writer, reader, {});
  assert(Contains(table.encoded_relation, table.linear_relation));
  assert(Contains(table.linear_relation, table.encoded_relation));
  Points shared;
  for (auto const& [task, element] : expected_reads) {
    auto image = element[0], channel = element[1], ih = element[2], iw = element[3];
    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
    auto pair = task;
    auto source = Coordinate(writer, {(image * h * w + ih * w + iw) / producer_m,
                                      channel / producer_c});
    pair.insert(pair.end(), source.begin(), source.end());
    shared.insert({pair, element});
  }
  assert(PointsOf(exact.shared_elements) == shared);
  assert(exact.metrics.volume.SumDomain().Eval({}) == long(shared.size()));
  assert(exact.metrics.wait.SumDomain().Eval({}) == long(expected_edges.size()));
  assert(exact.metrics.count.Eval({}) == reader.Count().Eval({}, {}));
  {
    producer.name = "producer"; producer.exact_task_access = true;
    producer.task_space = writer.output; producer.task_map = wp.ownership;
    consumer.name = "consumer"; consumer.exact_task_access = true;
    consumer.task_space = reader.output; consumer.task_map = rp.ownership;
    if (split) consumer.task_space.axes.pop_back();
    consumer.result = {"output", {Axis("n", n), Axis("c", o), Axis("h", p), Axis("w", q)}};
    consumer.result_map.results = {I("m", 1, p * q), I("oc"),
        Sum({I("m", 1, q), I("m", -p, p * q)}), Sum({I("m"), I("m", -q, q)})};
    consumer.operands.push_back({producer.name, halo, map, {}});
    assert(DecodeSemanticOp(EncodeSemanticOp(consumer)).Serialize() == consumer.Serialize());
    Granularity granularity;
    granularity.Tile(producer.name, "m", F(producer_m)).Tile(producer.name, "c", F(producer_c));
    granularity.Tile(consumer.name, "m", F(tm)).Tile(consumer.name, "oc", F(tn));
    if (split) {
      consumer.reduction.partial_tensor = "partials"; consumer.reduction.combiner = "consumer.combine";
      granularity.Split(consumer.name, F(split));
    }
    auto graph = Instantiate({{producer, consumer}}, granularity);
    auto const& lowered = graph.nodes[1].operands[0];
    assert(lowered.axes[2].kind == OperandAxisMap::Kind::kIndexed);
    assert(std::any_of(lowered.axes[2].terms.begin(), lowered.axes[2].terms.end(),
        [](auto const& term) { return term.window_dim == "r"; }));
    auto edges = CouplingDerivation{}.Derive(graph, {});
    assert(edges.size() == (split ? 2u : 1u) && edges[0].tier == Tier::kAffine && edges[0].exact);
    Check(edges[0].C, expected_edges);
    assert(edges[0].shared_elements && edges[0].interface_elements);
    assert(edges[0].consumer_elements && edges[0].read_box);
    assert(Contains(*edges[0].read_box, *edges[0].consumer_elements));
    assert(PointsOf(*edges[0].shared_elements) == shared);
    assert(edges[0].interface_elements->Eval({}) >= 0);
    auto work = DeriveTaskWork(consumer, graph.nodes[1], {});
    assert(work.read_elements.SumDomain().Eval({}) == long(expected_reads.size()));
    assert(work.write_elements.SumDomain().Eval({}) == n * p * q * o * (split ? (c + split - 1) / split : 1));
    assert(work.reduce_extent.Eval({}) == r * s * (channelwise ? 1 : c));
    auto envelope = TaskElementBoxEnvelope(reads);
    assert(Contains(envelope, reads));
  }
}
void PixelShuffle() {
  constexpr int n = 2, h = 3, w = 5, c = 3, r = 2, tm = 7, tn = 5;
  SemanticOp producer;
  producer.domain = {Dim("m", n * h * w), Dim("c", c * r * r)};
  TensorSpace tensor{"shuffle", {Axis("n", n), Axis("c", c), Axis("h", h * r), Axis("w", w * r)}};
  producer.result = tensor;
  producer.result_map.results = {I("m", 1, h * w), I("c", 1, r * r),
      Sum({I("m", r, w), I("m", -r * h, h * w), I("c", 1, r), I("c", -r, r * r)}),
      Sum({I("m", r), I("m", -r * w, w), I("c"), I("c", -r, r)})};
  OperatorNode writer;
  writer.output = {"ownership", {Axis("m", n * h * w), Axis("c", c * r * r)}};
  writer.tile = {F(tm), F(tn)};
  TaskElementPartition partition; partition.ownership.results = {I("m"), I("c")};
  auto writes = ProjectTaskElements(producer, writer, partition, tensor, producer.result_map, {});
  assert(writes.Reverse().IsSingleValued());
  assert(PointsOf(writes).size() == n * h * w * c * r * r);
  SemanticOp consumer;
  consumer.domain = {Dim("m", n * h * w * r * r), Dim("c", c)};
  IndexingMap identity{{I("m", 1, h * w * r * r), I("c"),
      Sum({I("m", 1, w * r), I("m", -h * r, h * w * r * r)}),
      Sum({I("m"), I("m", -w * r, w * r)})}};
  OperatorNode reader;
  reader.output = {"ownership", {Axis("m", n * h * w * r * r), Axis("c", c)}};
  reader.tile = {F(9), F(2)};
  auto reads = ProjectTaskElements(consumer, reader, partition, tensor, identity, {});
  Points expected;
  for (int image = 0; image < n; ++image) for (int ph = 0; ph < h * r; ++ph)
    for (int pw = 0; pw < w * r; ++pw) for (int channel = 0; channel < c; ++channel) {
      int pixel = image * h * w * r * r + ph * w * r + pw;
      int input_pixel = image * h * w + (ph / r) * w + pw / r;
      int input_channel = channel * r * r + (ph % r) * r + pw % r;
      expected.insert({Coordinate(reader, {pixel / 9, channel / 2}),
                       Coordinate(writer, {input_pixel / tm, input_channel / tn})});
    }
  auto coupling = reads.ApplyRange(writes.Reverse());
  Check(coupling, expected);
  auto table = BuildDependencyTable(coupling, writer, reader, {});
  assert(table.stride > 1);
}
void PeriodicEnvelope() {
  auto reads=CouplingRelation::FromIslText(
      "{ [m] -> [h] : exists (e0: 0 <= m <= 32767 and 0 <= h <= 255 and "
      "0 <= e0 <= 255 and 4m - h - 256e0 <= 65536*floor((m-64e0)/16384) "
      "<= 3 + 4m - h - 256e0) }");
  auto expected=CouplingRelation::FromIslText(
      "{ [m] -> [h] : 0 <= m <= 32767 and "
      "4m - 256*floor(m/64) <= h <= 3 + 4m - 256*floor(m/64) }");
  auto box=TaskElementBoxEnvelope(reads);
  assert(Contains(box,expected) && Contains(expected,box));
  auto holes=CouplingRelation::FromIslText(
      "[S] -> { [m] -> [h,w] : 0 <= m < S and (m mod 3)=0 and "
      "(h=m or h=m+4) and (w=2m or w=2m+3) }");
  auto filled=CouplingRelation::FromIslText(
      "[S] -> { [m] -> [h,w] : 0 <= m < S and (m mod 3)=0 and "
      "m <= h <= m+4 and 2m <= w <= 2m+3 }");
  auto hole_box=TaskElementBoxEnvelope(holes);
  assert(Contains(hole_box,filled) && Contains(filled,hole_box));
}
}
int TestWindowTaskAccess(int, char**) {
  IslContext context;
  PeriodicEnvelope();
  Window(2, 5, 7, 5, 3, 3, 3, 1, 1, 1, false, 9, 2);
  Window(2, 5, 7, 5, 3, 3, 3, 2, 1, 1, false, 5, 2);
  Window(1, 7, 9, 5, 3, 3, 3, 1, 2, 2, false, 5, 2);
  Window(1, 7, 9, 3, 5, 7, 7, 2, 3, 1, false, 7, 3);
  Window(2, 6, 8, 5, 3, 2, 2, 2, 0, 1, false, 5, 2);
  Window(2, 5, 7, 5, 5, 3, 3, 2, 1, 1, true, 5, 2);
  Window(2, 5, 7, 5, 5, 3, 3, 1, 1, 1, true, 9, 2);
  Window(2, 5, 7, 5, 3, 3, 3, 2, 1, 1, false, 5, 2, 2);
  Window(1, 5, 7, 3, 3, 3, 3, 1, 1, 1, false, 9, 2, 3);
  PixelShuffle();
  return 0;
}
}
