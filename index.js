import 'dotenv/config';
import { webcrypto as crypto } from 'node:crypto';

globalThis.crypto = crypto;

import * as z from 'zod';
import { tool } from '@langchain/core/tools';
import { ChatOpenRouter } from '@langchain/openrouter';
import { HumanMessage } from '@langchain/core/messages';

const multiply = tool(
    async ({ a, b }) => {
        console.log(`[tool called] multiply(${a}, ${b})`);
        return String(a * b);
    },
    {
        name: 'multiply',
        description: 'Multiply two numbers together.',
        schema: z.object({
            a: z.number().describe('The first number'),
            b: z.number().describe('The second number'),
        }),
    }
);

const subtract = tool(
    async ({ a, b }) => {
        console.log(`[tool called] subtract(${a}, ${b})`);
        return String(a - b);
    },
    {
        name: 'subtract',
        description: 'subtract two numbers from one another.',
        schema: z.object({
            a: z.number().describe('The first number'),
            b: z.number().describe('The second number'),
        }),
    }
);

const convertFtC = tool(
    async ({ a }) => {
        console.log(`[tool called] Convert Fahrenheit to Celsius(${a})`);
        return String((a - 32) * (5 / 9));
    },
    {
        name: 'convertFtC',
        description: 'Convert from Fahrenheit to Celsius USING THE SUBTRACT AND MULTIPLY TOOL.',
        schema: z.object({
            a: z.number().describe('The temperature in Fahrenheit'),
        }),
    }
);

const getCoordinates = tool(
    async ({ location }) => {
        console.log(`[tool called] getCoordinates(${location})`);
        if (location.toLowerCase().includes('santa cruz')) {
            return JSON.stringify({ latitude: 36.9741, longitude: -122.0308 });
        }
        return JSON.stringify({ latitude: 0, longitude: 0 });
    },
    {
        name: 'getCoordinates',
        description: 'Get latitude and longitude for a given city/location string.',
        schema: z.object({
            location: z.string().describe('The city and state/country, e.g., "Santa Cruz, CA"'),
        }),
    }
);

const getWeather = tool(
    async ({ latitude, longitude }) => {
        console.log(`[tool called] getWeather(${latitude}, ${longitude})`);
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,wind_speed_10m&temperature_unit=fahrenheit`;
        const res = await fetch(url);
        const data = await res.json();
        const current = data.current;
        return `Temperature: ${current.temperature_2m}°F, Wind: ${current.wind_speed_10m} mph`;
    },
    {
        name: 'get_weather',
        description: 'Get the current temperature and wind speed for a location.',
        schema: z.object({
            latitude: z.number().describe('Latitude'),
            longitude: z.number().describe('Longitude'),
        }),
    }
);

const model = new ChatOpenRouter({
    model: "meta-llama/llama-3.1-70b-instruct",
    apiKey: process.env.OPENROUTER_API_KEY,
});

const tools = [multiply, subtract, getCoordinates, getWeather, convertFtC];
const toolMap = Object.fromEntries(tools.map(t => [t.name, t]));
const modelWithTools = model.bindTools(tools, { parallel_tool_calls: false });

(async () => {
    const messages = [new HumanMessage(
        `You must execute your thoughts strictly step-by-step. Use getCoordinates to look up "Santa Cruz, CA". Stop and wait for the tool output. + Step 2: Use those coordinates to call getWeather. Stop and wait for the tool output. Step 3: Take that Fahrenheit temperature and convert it using convertFtC .`
    )];

    var response = await modelWithTools.invoke(messages);
    let loopCount = 0;

    while (response.tool_calls && response.tool_calls.length > 0) {
        if (loopCount++ > 5) {
            console.log('\n[Guardrail triggered: Stopped potential infinite tool loop]');
            break;
        }

        messages.push(response);
        
        for (const toolCall of response.tool_calls) {
            const selectedTool = toolMap[toolCall.name];
            if (!selectedTool) throw new Error(`Unknown tool: ${toolCall.name}`);
            
            console.log(`\n[Executing tool: ${toolCall.name}]`);
            try {
                const toolMessage = await selectedTool.invoke(toolCall);
                messages.push(toolMessage);
            } catch (err) {
                console.log(`[Tool Error] Could not complete ${toolCall.name}`);
            }
        }
        
        response = await modelWithTools.invoke(messages);
    }

    messages.push(response);

    process.stdout.write('\n ');
    const stream = await modelWithTools.streamEvents(messages, { version: 'v2' });

    for await (const event of stream) {
        if (event.event === 'on_chat_model_stream') {
            const chunk = event.data?.chunk?.content || event.data?.chunk?.text;
            if (chunk) process.stdout.write(chunk);
        }
    }
    console.log('Final answer:', response.content);
})();