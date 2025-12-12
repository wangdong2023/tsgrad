

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

export function mseLoss(target: Value[], pred: Value[]): Value {
    return Array(target.length).fill(0)
        .map((v, i) => pred[i].minus(target[i]))
        .map(v => v.mul(v))
        .reduce((p, c, ci) => p.add(c))
        .mul(1 / target.length);
}


export class Value { // Kapathy implements more, but I'm lazy, this should do
    public grad: number = 0;
    public _backward: () => void = () => {};

    constructor(
        public data: number,
        public prev: Value[] =[],
        public label: string = '_') {
    }

    add(other: Value | number): Value {
        if (typeof other === 'number') {
            other = new Value(other);
        }

        let out = new Value(this.data + other.data, [this, other], `(${this.label}+${other.label})`);
        out._backward = () => {
            this.grad += out.grad;
            other.grad += out.grad;
        }
        return out;
    }

    minus(other: Value | number): Value { // different choice, we want toString to be prettier, instead a+ -b
        if (typeof other === 'number') {
            other = new Value(other);
        }
        let out = new Value(this.data - other.data, [this, other], `(${this.label}-${other.label})`);
        out._backward = () => {
            this.grad += out.grad;
            other.grad -= out.grad;
        }

        return out;
    }

    mul(other: Value | number): Value {
        if (typeof other === 'number') {
            other = new Value(other);
        }
        let out = new Value(this.data * other.data, [this, other], `${this.label}*${other.label}`);
        out._backward = () => {
            this.grad += out.grad * other.data;
            other.grad += out.grad * this.data;
        }

        return out;
    }

    tanh(): Value {
        let ev = Math.pow(Math.E, 2 * this.data);
        let out = new Value((ev - 1) / (ev + 1), [this], `tanh(${this.label})`);
        out._backward = () => {
            this.grad += out.grad * (1 - Math.pow(out.data, 2));
        }
        return out;
    }

    toString(precision: number=3): string {
        return `${this.label} {value:${this.data.toPrecision(precision)}, grad:${this.grad.toPrecision(precision)}}`
    }

    backward() {
        let params = topoSort(this, new Set([]));
        params.reverse();
        for (const param of params) {
            // console.log("Before ", param.toString());
            param._backward();
            // console.log("After ", param.toString());
        }
    }
}

function random(): number {
    return Math.random() * 2 - 1;
}

export function createArray(n: number, label: string='w'): Value[] {
    return Array(n).fill(0).map(
        (_, i) => new Value(random(), [], `${label}${i}`));
}

export function valueArray(vals: number[], label: string='v'): Value[] {
    return vals.map(
        (v, i) => new Value(v, [], `${label}${i}`))
}


export class Neuron {
    public params: Value[];
    public bias: Value;

    constructor(public n: number) {
        this.params = createArray(n);
        this.bias = new Value(random(), [], 'bias');
    }

    get_parameters(): Value[] {
        return [...this.params, this.bias];
    }

    forward(input: Value[]): Value { // a bit uglier
        let sum = this.bias;
        for (let i = 0; i < this.n; ++i) {
            let temp = this.params[i].mul(input[i]);
            sum = sum.add(temp);
        }
        let out = sum.tanh();
        return out;
    }
}
