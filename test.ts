import { createArray, mseLoss, Neuron, Value, valueArray } from ".";

export function assert(condition: unknown, msg?: string): asserts condition {
  if (!condition) {
    throw new Error(msg || 'Assertion failed');
  }
}

function test_basic() {
    let x = new Value(1, [], 'x');
    let x1 = x.add(1);
    let y = new Value(3, [], 'y');
    let b = new Value(1, [], 'b');
    let xy = x1.mul(y);
    let z = xy.add(b); // z = x * y + b
    z.label = 'z';

    z.grad = 1;
    z._backward();

    b._backward();
    xy._backward();
    x1._backward();

    x._backward();

    assert(z.grad == 1);
    assert(b.grad == 1);
    assert(xy.grad == 1);
    assert(x.grad == 3);
    assert(y.grad == 2);
    console.log("test_basic passed!")
}


function test_backward() {
    let x = new Value(1, [], 'x');
    let x1 = x.add(1);
    let y = new Value(3, [], 'y');
    let b = new Value(1, [], 'b');
    let xy = x1.mul(y);
    let z = xy.add(b); // z = x * y + b
    z.label = 'z';

    z.grad = 1;
    z.backward()

    assert(z.grad == 1);
    assert(b.grad == 1);
    assert(xy.grad == 1);
    assert(x.grad == 3);
    assert(y.grad == 2);
    console.log("test_backward passed!")
}

const testBackwardNeuron = function test_backward_neuron() {
  let x = createArray(3, 'x');

  let nu = new Neuron(3);

  let o = nu.forward(x);
  let o1 = o.data
  o.grad = 1;
  o.backward();

  for(const v of nu.get_parameters()) {
      v.data = v.data + 0.01 * v.grad;
  }

  o = nu.forward(x);

  assert(o.data - o1 > 0);
  console.log(`${testBackwardNeuron.name} passed`);
}


function testMse() {
    let target = valueArray([0, 1]);
    let pred = valueArray([1, 1]);

    let mse = mseLoss(target, pred);
    assert(mse.data == 0.5);
}

test_basic();
test_backward();
testBackwardNeuron();
testMse();