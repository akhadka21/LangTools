import 'dotenv/config';
import * as z from 'zod';
import { tool } from '@langchain/core/tools';
import { ChatOpenRouter } from '@langchain/openrouter';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';

// --- Define a tool ---
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

// --- Set up the model and bind the tool ---
const model = new ChatOpenRouter({
    model: 'meta-llama/llama-3.3-70b:free',
    apiKey: process.env.OPENROUTER_API_KEY,
});

const modelWithTools = model.bindTools([multiply]);

// --- Send a message that should trigger the tool ---
const messages = [new HumanMessage('What is 47 multiplied by 83?')];
const response = await modelWithTools.invoke(messages);

console.log('Tool calls requested:', response.tool_calls);
messages.push(response);  // add AI's tool-call message to history

 // --- Execute the tool call(s) the model requested ---
for (const toolCall of response.tool_calls) {
    const result = await multiply.invoke(toolCall);
    messages.push(result);
}

// --- Send the tool result back to get the final answer ---
const finalResponse = await modelWithTools.invoke(messages);
console.log('Final answer:', finalResponse.content);