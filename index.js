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
        description: 'Convert from Fahrenheit to Celsius',
        schema: z.object({
            a: z.number().describe('The temperature in Fahrenheit'),
        }),
    }
);

const getCoordinates = tool(
    async ({ location }) => {
        console.log(`[tool called] getCoordinates(${location})`);

        const buildQueries = (raw) => {
            const queries = new Set();
            const cleaned = raw.trim();
            if (!cleaned) return [];

            queries.add(cleaned);
            queries.add(cleaned.replace(/,/g, ' ').replace(/\s+/g, ' ').trim());
            queries.add(cleaned.replace(/[,.-]/g, ' ').replace(/\s+/g, ' ').trim());

            const parts = cleaned.split(',').map((part) => part.trim()).filter(Boolean);
            if (parts.length > 1) {
                queries.add(parts.join(' '));
                queries.add(parts[0]);
                queries.add(`${parts[0]} ${parts[1]}`);
            }

            return [...queries].filter(Boolean);
        };

        const formatResult = (result, source) => {
            const name = result.name || result.display_name || 'Unknown location';
            const latitude = result.latitude ?? result.lat;
            const longitude = result.longitude ?? result.lon;
            const country = result.country || result.address?.country || '';
            const admin1 = result.admin1 || result.address?.state || result.address?.city || '';
            const locationLabel = [name, admin1, country].filter(Boolean).join(', ');
            return `Source: ${source} | Latitude: ${latitude}, Longitude: ${longitude}, Location: ${locationLabel}`;
        };

        const tryOpenMeteo = async (query) => {
            const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=3&language=en&format=json`;
            const res = await fetch(url);
            if (!res.ok) return null;
            const data = await res.json();
            return data.results?.[0] || null;
        };

        const tryNominatim = async (query) => {
            const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(query)}`;
            const res = await fetch(url, {
                headers: {
                    'Accept-Language': 'en',
                    'User-Agent': 'langtools/1.0',
                },
            });
            if (!res.ok) return null;
            const data = await res.json();
            if (!data?.[0]) return null;
            return {
                name: data[0].display_name,
                latitude: parseFloat(data[0].lat),
                longitude: parseFloat(data[0].lon),
                country: data[0].address?.country,
                address: data[0].address,
            };
        };

        for (const query of buildQueries(location)) {
            const openMeteoResult = await tryOpenMeteo(query);
            if (openMeteoResult) {
                return formatResult(openMeteoResult, 'Open-Meteo');
            }

            const nominatimResult = await tryNominatim(query);
            if (nominatimResult) {
                return formatResult(nominatimResult, 'Nominatim');
            }
        }

        return `No coordinates found for "${location}".`;
    },
    {
        name: 'getCoordinates',
        description: 'Get latitude and longitude for any city, town, or location string.',
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
        `You must execute your thoughts strictly step-by-step. Use getCoordinates to look up "San Leandro, CA". Stop and wait for the tool output. Use those coordinates to call getWeather. Stop and wait for the tool output. Take that Fahrenheit temperature and convert it using convertFtC .`
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