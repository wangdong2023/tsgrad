
class Value {
    public grad: number = 0;
    public _backward: () => void = () => {};

    constructor(
        public data: number,
        public prev: Value[] =[],
        public label: string = '',
        public ops: string = '') {
    }

    add(other: Value): Value {
        let out = new Value(this.data + other.data, [this, other], '', '+');
        this._backward = () => {
            this.grad += out.grad;
            other.grad += out.grad;
        }
        return out;
    }

    mul(other: Value): Value {
        let out = new Value(this.data * other.data, [this, other], '', '*');
        this._backward = () => {
            // console.log(out, this, other);
            this.grad += out.grad * other.data;
            other.grad += out.grad * this.data;
        }

        return out;
    }

    toString(): string {
        return `${this.label}(${this.data}, ${this.grad})`
    }
}



let x = new Value(2, [], 'x');
let y = new Value(3, [], 'y');
let b = new Value(1, [], 'b');
let xy = x.mul(y);
let z = xy.add(b); // z = x * y + b
z.label = 'z';
console.log(z.toString());
console.log(x.toString());
console.log(y.toString());

z.grad = 1;
z._backward();

b._backward();
console.log(x.toString());
console.log(y.toString());

xy._backward();

x._backward();
console.log(x.toString());
console.log(y.toString());
console.log(b.toString());
console.log(xy.toString());

