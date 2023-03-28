# Template Deep Learning Framework with CUDA/CUDNN support

## Creating a Tensor

Defining the shape of the tensor

```
Shape shape(510, 10, 10, 1);
```

setup the tensor on the GPU 
```
Tensor<gpu> tensor(shape);
```

or on the cpu

```
Tensor<cpu> tensor(shape);
```

allocate the memory on the gpu/cpu

```
tensor.allocate();
```

Initilize the tensor

```
tensor = Initilizer< Constant<DEVICE, float> >(shape, val);
```

```
tensor = Initilizer<Gaussian<DEVICE, DEFAULT_STREAM, float> >(shape, val, 2); // mu, std
```

## Perform aritimetics on tensors

```
Scalar<DEVICE> scalar(1);

auto tensorRes = ( scalar + (tensorA + tensorB - tensorC*tensorD) / tensorA).eval();
```

all steps are chained and used to build a computational graph on one of the steams (when we are doing runs in parallell), is only after we call eval() that we calculate the resulting tensor.


## Perform elementwise transformations on the tensor

Define the transformation

```
struct Test{
TENSOR_INLINE static float Eval(float v) { return 2.2;}
};
```

```
Tensor<DEVICE> tensorA(shape); tensorA.allocate();

tensorA = Initilizer< Constant<DEVICE, float> >(shape, 1.1);

auto tensorRes = F<Test>(tensorA).eval();
```

## Save tensor and load back

to save
```
tensor.write("test.dat");
```

and to read 

```
tensor_loaded.read("test.dat");
```

## Cast a tensor


```
Tensor<DEVICE, 0, float> tensor(shape); tensor.allocate();
tensor = Initilizer<Constant<DEVICE, float> >(shape, 2.6);
Tensor<DEVICE, 0, uint8_t> tensor8(shape); tensor8.allocate();

tensor8 = cast<uint8_t>(tensor);
```


## Perform Linear Algebra with the tensors 

Here we perform matrix multiplication between tensorA and a transposed version of tensorB
```
tensorC = linalg::matmul(tensorA, linalg::transpose(tensorB));
```


## Concate two tensors along an axis

```
auto tensor = operators::concat(tensor1, tensor2, 3);
```

## Permute the tensor - change the axis

```
std::vector<index_t> new_order = {2, 3, 0, 1};
auto tensor_transformed = operators::permute(tensor, new_order);
```


## Neural Network Primitives

### Fully connected layers

Setting up two layers, one fully connected and one bias layer, initialize the weights and biases

```
nn::FullyConnected<DEVICE> fully(3, 4);
nn::Bias<DEVICE> bias(1, 4);

auto cweight = fully.param("weight")->data();
auto cbias = bias.param("weight")->data();

float biasf = 0.2;

initilize::gaussian(cweight, 0, 1);
initilize::constant(cbias, (float)biasf);
```

checkout https://github.com/johannes-skog/projectx/blob/dev/tests/modules/nn_layers.test.cpp for how we setup conv-layers and how they are stacked and how we are testing that the gradient backward pass is working