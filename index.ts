

function topoSort(val: Value, seen: Set<Value>): Value[] {
    let out: Value[] = [];

    for (const child of val.prev) {
        if (seen.has(child)) {
            continue;
        }
        out = out.concat(topoSort(child, seen));
        seen.add(child);
    }

    out.push(val);

    return out;
}


export class Value { // Kapathy implements more, but I'm lazy, this should do
    public grad: number = 0;
    public _backward: () => void = () => {};

    constructor(
        public data: number,
        public prev: Value[] =[],
        public ops: string = '',
        public label: string = '_') {
    }

    add(other: Value | number): Value {
        if (typeof other === 'number') {
            other = new Value(other);
        }

        let out = new Value(this.data + other.data, [this, other], '+', '_');
        this._backward = () => {
            this.grad += out.grad;
            other.grad += out.grad;
        }
        return out;
    }

    mul(other: Value | number): Value {
        if (typeof other === 'number') {
            other = new Value(other);
        }
        let out = new Value(this.data * other.data, [this, other], '*', '_');
        this._backward = () => {
            // console.log(out, this, other);
            this.grad += out.grad * other.data;
            other.grad += out.grad * this.data;
        }

        return out;
    }

    tanh(): Value {
        let ev = Math.pow(Math.E, 2 * this.data);
        let out = new Value((ev - 1) / (ev + 1), [this], 'tanh', '_');
        this._backward = () => {
            this.grad += out.grad * (1 - Math.pow(out.data, 2));
        }
        return out;
    }

    toString(precision: number=3): string {
        return `${this.label}(${this.data.toPrecision(precision)}, ${this.grad.toPrecision(precision)})`
    }

    backward() {
        let params = topoSort(this, new Set([]));
        params.reverse();
        for (const param of params) {
            param._backward();
        }

    }
};


let x = new Value(2);
let o = x.tanh();
o.grad = 1;
x._backward()
console.log(x.toString());


