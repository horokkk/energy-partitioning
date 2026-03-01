// server.js - Node.js Express server (Non-AI workload)
// Usage: node server.js
// Load test: autocannon -c 100 -d 120 http://localhost:3000

const express = require('express');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = 3000;

// Access log setup
const LOG_DIR = process.env.LOG_DIR || '.';
const logPath = path.join(LOG_DIR, 'nodejs_access.log');
const logStream = fs.createWriteStream(logPath, { flags: 'a' });
console.log(`Access log: ${logPath}`);

// Logging middleware
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const ms = Date.now() - start;
        logStream.write(`${new Date().toISOString()} ${req.method} ${req.url} ${res.statusCode} ${ms}ms\n`);
    });
    next();
});

// Endpoints that generate CPU load

// Default route
app.get('/', (req, res) => {
    // JSON serialize/deserialize + hash computation (CPU load)
    const data = {
        timestamp: Date.now(),
        random: Math.random(),
        items: Array.from({ length: 100 }, (_, i) => ({
            id: i,
            value: crypto.randomBytes(32).toString('hex')
        }))
    };

    const serialized = JSON.stringify(data);
    const hash = crypto.createHash('sha256').update(serialized).digest('hex');

    res.json({
        hash: hash,
        size: serialized.length,
        itemCount: data.items.length
    });
});

// Fibonacci computation (CPU intensive)
app.get('/compute', (req, res) => {
    const n = 35;
    function fib(n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
    const result = fib(n);
    res.json({ fibonacci: n, result: result });
});

// Sorting (memory + CPU)
app.get('/sort', (req, res) => {
    const arr = Array.from({ length: 10000 }, () => Math.random());
    arr.sort((a, b) => a - b);
    res.json({ sorted: arr.length, first: arr[0], last: arr[arr.length - 1] });
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log('Endpoints: /, /compute, /sort');
    console.log('Load test: autocannon -c 100 -d 120 http://localhost:3000');
});
