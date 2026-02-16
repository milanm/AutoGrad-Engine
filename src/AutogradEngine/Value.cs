// =============================================================================
// PART 1: THE AUTOGRAD ENGINE
// =============================================================================
//
// The big question in neural networks: "If I change this one number a tiny bit,
// how much does the overall error change?" The answer is called a "gradient."
//
// The Value class wraps every number and secretly records every math operation
// done to it — building a chain. When you call Backward(), it walks backward
// through that chain to figure out each number's influence on the final error.
//
// This technique is called "backpropagation." It's the core of how neural
// networks learn. In production, PyTorch or TensorFlow does this on tensors
// (big arrays of numbers) on GPUs. Here we do it one scalar at a time.
//
// Think of it like a spreadsheet: if cell Z depends on Y, which depends on X,
// and you change X by 0.001, how much does Z change? The chain rule from
// calculus answers this automatically for any chain of operations.
// =============================================================================

using System;
using System.Collections.Generic;

public class Value
{
    // The actual number (e.g. 0.37, -1.2, etc.)
    public double Data;

    // The gradient: `∂finalLoss/∂this` or "how much does the final error change if I nudge this number?"
    // Starts at 0, gets filled in during Backward().
    public double Grad;

    // A function that computes this node's contribution to gradients.
    // Each operation (+, *, exp, etc.) defines its own gradient rule.
    private Action _backward;

    // The Values that were inputs to the operation that created this Value.
    // This is how we track the computation graph — like a family tree of math.
    private readonly HashSet<Value> _prev;

    // What operation created this node ("+", "*", "exp", etc.) — for debugging.
    public string Op { get; }

    public Value(double data, IEnumerable<Value>? children = null, string op = "")
    {
        Data = data;
        Grad = 0;
        _backward = () => { };
        _prev = children != null ? new HashSet<Value>(children) : new HashSet<Value>();
        Op = op;
    }

    // --- Operator overloads: each one does two things ---
    // 1. Computes the forward result (normal math)
    // 2. Defines the backward rule (how gradients flow back through this operation)

    // Addition: d(a+b)/da = 1, d(a+b)/db = 1
    // Both inputs get the full gradient from the output.
    public static Value operator +(Value a, Value b)
    {
        var outVal = new Value(a.Data + b.Data, new[] { a, b }, "+");
        outVal._backward = () =>
        {
            a.Grad += outVal.Grad;
            b.Grad += outVal.Grad;
        };
        return outVal;
    }

    public static Value operator +(Value a, double b) => a + new Value(b);
    public static Value operator +(double a, Value b) => new Value(a) + b;

    // Multiplication: `d(finalLoss)/d(a) = d(finalLoss)/d(a*b) * d(a*b)/da` and `d(a*b)/da = b`. So `a.Grad = outVal.Grad * b`
    // Each input's gradient is scaled by the OTHER input's value.
    // Intuition: if b is large, a small change in a has a big effect on the product.
    public static Value operator *(Value a, Value b)
    {
        var outVal = new Value(a.Data * b.Data, new[] { a, b }, "*");
        outVal._backward = () =>
        {
            a.Grad += b.Data * outVal.Grad;
            b.Grad += a.Data * outVal.Grad;
        };
        return outVal;
    }

    public static Value operator *(Value a, double b) => a * new Value(b);
    public static Value operator *(double a, Value b) => new Value(a) * b;

    // Other arithmetic: defined in terms of + and * so they get gradients for free.
    public static Value operator -(Value a) => a * -1;
    public static Value operator -(Value a, Value b) => a + (-b);
    public static Value operator -(Value a, double b) => a + (-b);
    public static Value operator -(double a, Value b) => a + (-b);
    public static Value operator /(Value a, Value b) => a * b.Pow(-1);
    public static Value operator /(Value a, double b) => a * new Value(b).Pow(-1);
    public static Value operator /(double a, Value b) => new Value(a) * b.Pow(-1);

    // Power: d(x^n)/dx = n * x^(n-1) — the classic calculus power rule.
    public Value Pow(double exponent)
    {
        var outVal = new Value(Math.Pow(Data, exponent), new[] { this }, $"**{exponent}");
        outVal._backward = () =>
        {
            Grad += exponent * Math.Pow(Data, exponent - 1) * outVal.Grad;
        };
        return outVal;
    }

    // Log: d(ln(x))/dx = 1/x — used in computing the loss function.
    public Value Log()
    {
        var outVal = new Value(Math.Log(Data), new[] { this }, "log");
        outVal._backward = () =>
        {
            Grad += (1.0 / Data) * outVal.Grad;
        };
        return outVal;
    }

    // Exp: d(e^x)/dx = e^x — the exponential is its own derivative.
    // Used in softmax to convert raw scores into probabilities.
    public Value Exp()
    {
        var outVal = new Value(Math.Exp(Data), new[] { this }, "exp");
        outVal._backward = () =>
        {
            Grad += outVal.Data * outVal.Grad;
        };
        return outVal;
    }

    // ReLU (Rectified Linear Unit): max(0, x)
    // The simplest activation function. If x > 0, pass it through. If x < 0, output 0.
    // Gradient: 1 if x > 0, 0 if x < 0. Acts like a gate.
    public Value ReLU()
    {
        var outVal = new Value(Data < 0 ? 0 : Data, new[] { this }, "ReLU");
        outVal._backward = () =>
        {
            Grad += (outVal.Data > 0 ? 1.0 : 0.0) * outVal.Grad;
        };
        return outVal;
    }

    // Backward(): The heart of learning.
    //
    // Step 1: Build a topological ordering of all Value nodes.
    //         (If A feeds into B which feeds into C, we need to process C, then B, then A.)
    //
    // Step 2: Walk backward through this ordering, applying the chain rule at each node.
    //         Each node's _backward() function pushes gradients to its inputs.
    //
    // After this runs, every Value in the graph knows its gradient — i.e., how much
    // the final error would change if you nudged that number up a tiny bit.
    public void Backward()
    {
        // Iterative topological sort. (The recursive version would overflow the
        // C# stack — the computation graph has thousands of nodes.)
        var topo = new List<Value>();
        var visited = new HashSet<Value>();
        var stack = new Stack<(Value node, bool processed)>();
        stack.Push((this, false));

        while (stack.Count > 0)
        {
            var (v, processed) = stack.Pop();
            if (processed) { topo.Add(v); continue; }
            if (visited.Contains(v)) continue;
            visited.Add(v);
            stack.Push((v, true));
            foreach (var child in v._prev)
                if (!visited.Contains(child))
                    stack.Push((child, false));
        }

        // The loss node gets gradient 1.0 (the starting point of backprop).
        // Then we propagate backward through every operation.
        Grad = 1.0;
        for (int i = topo.Count - 1; i >= 0; i--)
            topo[i]._backward();
    }

    public override string ToString() => $"Value(data={Data}, grad={Grad})";
}
