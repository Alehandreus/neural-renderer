#include "mesh.h"

#include <cfloat>
#include <cmath>
#include <vector>

#include <bvh/v2/bvh.h>
#include <bvh/v2/bbox.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/vec.h>

namespace {

using Scalar = float;
using Vec3f = bvh::v2::Vec<Scalar, 3>;
using BBox = bvh::v2::BBox<Scalar, 3>;
using Node = bvh::v2::Node<Scalar, 3>;
using Bvh = bvh::v2::Bvh<Node>;

Vec3f toBvhVec(const Vec3& v) {
    return Vec3f(v.x, v.y, v.z);
}

}  // namespace

void Mesh::buildBvh() {
    if (!bvhDirty_) {
        return;
    }

    nodes_.clear();
    if (triangles_.empty()) {
        bvhDirty_ = false;
        boundsDirty_ = false;
        boundsMin_ = Vec3(0.0f, 0.0f, 0.0f);
        boundsMax_ = Vec3(0.0f, 0.0f, 0.0f);
        return;
    }

    std::vector<BBox> bboxes(triangles_.size());
    std::vector<Vec3f> centers(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); ++i) {
        const Triangle& tri = triangles_[i];
        Vec3f v0 = toBvhVec(tri.v0);
        Vec3f v1 = toBvhVec(tri.v1);
        Vec3f v2 = toBvhVec(tri.v2);
        BBox bbox(v0, v0);
        bbox.extend(v1);
        bbox.extend(v2);
        bboxes[i] = bbox;
        centers[i] = bbox.get_center();
    }

    bvh::v2::ThreadPool threadPool;
    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
    Bvh bvh = bvh::v2::DefaultBuilder<Node>::build(threadPool, bboxes, centers, config);

    std::vector<Triangle> reordered(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); ++i) {
        size_t src = bvh.prim_ids[i];
        reordered[i] = triangles_[src];
    }
    triangles_ = std::move(reordered);

    nodes_.resize(bvh.nodes.size());
    for (size_t i = 0; i < bvh.nodes.size(); ++i) {
        const Node& node = bvh.nodes[i];
        BvhNode out{};
        out.boundsMin = Vec3(node.bounds[0], node.bounds[2], node.bounds[4]);
        out.boundsMax = Vec3(node.bounds[1], node.bounds[3], node.bounds[5]);
        if (node.is_leaf()) {
            out.isLeaf = 1;
            out.first = static_cast<int>(node.index.first_id());
            out.count = static_cast<int>(node.index.prim_count());
            out.left = -1;
            out.right = -1;
        } else {
            out.isLeaf = 0;
            out.left = static_cast<int>(node.index.first_id());
            out.right = static_cast<int>(node.index.first_id() + 1);
            out.first = 0;
            out.count = 0;
        }
        nodes_[i] = out;
    }

    bvhDirty_ = false;
    deviceDirty_ = true;
    deviceNodesDirty_ = true;

    if (boundsDirty_) {
        Vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
        Vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (const Triangle& tri : triangles_) {
            min.x = fminf(min.x, fminf(tri.v0.x, fminf(tri.v1.x, tri.v2.x)));
            min.y = fminf(min.y, fminf(tri.v0.y, fminf(tri.v1.y, tri.v2.y)));
            min.z = fminf(min.z, fminf(tri.v0.z, fminf(tri.v1.z, tri.v2.z)));
            max.x = fmaxf(max.x, fmaxf(tri.v0.x, fmaxf(tri.v1.x, tri.v2.x)));
            max.y = fmaxf(max.y, fmaxf(tri.v0.y, fmaxf(tri.v1.y, tri.v2.y)));
            max.z = fmaxf(max.z, fmaxf(tri.v0.z, fmaxf(tri.v1.z, tri.v2.z)));
        }
        boundsMin_ = min;
        boundsMax_ = max;
        boundsDirty_ = false;
    }
}
