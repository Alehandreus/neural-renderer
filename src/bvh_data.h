#pragma once

#include "vec3.h"

struct BvhNode {
    Vec3 boundsMin;
    Vec3 boundsMax;
    int left;
    int right;
    int first;
    int count;
    int isLeaf;
};
