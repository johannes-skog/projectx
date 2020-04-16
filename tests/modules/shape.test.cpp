#include <catch.hpp>
#include <fundamental/macro.h>
#include <fundamental/shape.h>

#include <fundamental/transform/transform_slice.h>

using namespace  core;

TEST_CASE("Shape - deduced", "[shape]"){

  Shape shape(54, 2, 2, 3);
  index_t results[SHAPE_MAX_DIM] = {3, 1, 1, 1};
  index_t tesa[SHAPE_MAX_DIM];

  shape.deduce_shape(3, 10, tesa); // s = 3, i = 10

  SECTION("deduced"){
    for (int i = 0; i < shape.dim; ++i)
      REQUIRE(tesa[i] == results[i]);
  }

  SECTION("index"){
    index_t s, i;
    shape.index(s, i, tesa);
    REQUIRE(s == 3);
    REQUIRE(i == 10);
  }

}

TEST_CASE("Shape - reshaped", "[shape]"){

  Shape shape(54, 2, 2, 3);
  Shape reshaped = shape.reshape(54, 2, -1);

  REQUIRE(reshaped[0] == 54);
  REQUIRE(reshaped[1] == 2);
  REQUIRE(reshaped[2] == 6);

}

TEST_CASE("Shape - flatten", "[shape]"){

  Shape shape(54, 2, 2, 3);
  Shape flatten = shape.reshape(54, -1);

  REQUIRE(flatten[0] == 54);
  REQUIRE(flatten[1] == 12);

}

TEST_CASE("Shape - collapsed", "[shape]"){

  Shape shape(54, 2, 2, 3);
  Shape collaped = shape.collapse_to_dim(2);

  REQUIRE(collaped[0] == 54);
  REQUIRE(collaped[1] == 12);

}

TEST_CASE("Shape - sliced", "[shape]"){

  Shape shape(54, 20, 10, 300);

  SliceTransformer slice(shape, {0, -10,
                                 1, 8,
                                 1, 7,
                                 10, 255} );

  REQUIRE(slice.new_shape[0] == 44);
  REQUIRE(slice.new_shape[1] == 7);
  REQUIRE(slice.new_shape[2] == 6);
  REQUIRE(slice.new_shape[3] == 245);


}
