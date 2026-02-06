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

    bvhNodes_.clear();
    if (indices_.empty() || vertices_.empty()) {
        bvhDirty_ = false;
        boundsDirty_ = false;
        boundsMin_ = Vec3(0.0f, 0.0f, 0.0f);
        boundsMax_ = Vec3(0.0f, 0.0f, 0.0f);
        return;
    }

    // Build bounding boxes and centers for each triangle
    std::vector<BBox> bboxes(indices_.size());
    std::vector<Vec3f> centers(indices_.size());
    for (size_t i = 0; i < indices_.size(); ++i) {
        const uint3& idx = indices_[i];
        Vec3f v0 = toBvhVec(vertices_[idx.x]);
        Vec3f v1 = toBvhVec(vertices_[idx.y]);
        Vec3f v2 = toBvhVec(vertices_[idx.z]);
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

    // Reorder triangle indices according to BVH ordering
    std::vector<uint3> reorderedIndices(indices_.size());
    for (size_t i = 0; i < indices_.size(); ++i) {
        size_t src = bvh.prim_ids[i];
        reorderedIndices[i] = indices_[src];
    }
    indices_ = std::move(reorderedIndices);

    // Also need to remap materialMap to account for reordering
    // Build a mapping from old triangle index to new triangle index
    if (!materialMap_.empty() && !materialIds_.empty()) {
        // Create inverse mapping: newIdx -> oldIdx is bvh.prim_ids[newIdx]
        // We need oldIdx -> newIdx
        std::vector<size_t> oldToNew(indices_.size());
        for (size_t newIdx = 0; newIdx < bvh.prim_ids.size(); ++newIdx) {
            oldToNew[bvh.prim_ids[newIdx]] = newIdx;
        }

        // Rebuild materialMap with new triangle indices
        std::vector<uint32_t> newMaterialMap;
        std::vector<int> newMaterialIds;
        newMaterialMap.reserve(materialMap_.size());
        newMaterialIds.reserve(materialIds_.size());

        // For each original primitive, find where its triangles now start
        // This is complex because triangles from different primitives can be interleaved after BVH reorder
        // Instead, we assign material per-triangle and rebuild per-primitive grouping

        // Simpler approach: store material ID per triangle, then rebuild primitives
        std::vector<int> perTriMaterial(indices_.size(), -1);
        for (size_t prim = 0; prim < materialMap_.size(); ++prim) {
            uint32_t start = materialMap_[prim];
            uint32_t end = (prim + 1 < materialMap_.size()) ? materialMap_[prim + 1] : static_cast<uint32_t>(bvh.prim_ids.size());
            int matId = materialIds_[prim];
            for (uint32_t oldTri = start; oldTri < end; ++oldTri) {
                size_t newTri = oldToNew[oldTri];
                perTriMaterial[newTri] = matId;
            }
        }

        // Rebuild primitives: group consecutive triangles with same material
        int currentMat = perTriMaterial.empty() ? -1 : perTriMaterial[0];
        newMaterialMap.push_back(0);
        newMaterialIds.push_back(currentMat);

        for (size_t i = 1; i < perTriMaterial.size(); ++i) {
            if (perTriMaterial[i] != currentMat) {
                currentMat = perTriMaterial[i];
                newMaterialMap.push_back(static_cast<uint32_t>(i));
                newMaterialIds.push_back(currentMat);
            }
        }

        materialMap_ = std::move(newMaterialMap);
        materialIds_ = std::move(newMaterialIds);
    }

    // Convert BVH nodes to our format
    bvhNodes_.resize(bvh.nodes.size());
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
        bvhNodes_[i] = out;
    }

    bvhDirty_ = false;
    geometryDirty_ = true;
    bvhNodesDirty_ = true;
    materialsDirty_ = true;  // Material map changed

    // Compute bounds
    if (boundsDirty_) {
        Vec3 minV(FLT_MAX, FLT_MAX, FLT_MAX);
        Vec3 maxV(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (const Vec3& v : vertices_) {
            minV.x = fminf(minV.x, v.x);
            minV.y = fminf(minV.y, v.y);
            minV.z = fminf(minV.z, v.z);
            maxV.x = fmaxf(maxV.x, v.x);
            maxV.y = fmaxf(maxV.y, v.y);
            maxV.z = fmaxf(maxV.z, v.z);
        }
        boundsMin_ = minV;
        boundsMax_ = maxV;
        boundsDirty_ = false;
    }
}
