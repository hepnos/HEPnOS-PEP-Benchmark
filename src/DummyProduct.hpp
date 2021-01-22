#ifndef __DUMMY_PRODUCT_H
#define __DUMMY_PRODUCT_H

#include <vector>

struct dummy_product {
    std::vector<char> data;

    template<typename A>
    void serialize(A& ar, const unsigned int version) {
        ar & data;
    }

};

#endif
