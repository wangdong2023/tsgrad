
export class Value {
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
};





