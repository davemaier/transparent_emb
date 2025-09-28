#!/usr/bin/env node

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';
import fs from 'fs/promises';
import { parseArgs } from 'node:util';
import {
  modelRuns,
  featureVectors,
  generatedSegments,
  groundTruthSegments,
} from './drizzle_schema';
import { eq, and } from 'drizzle-orm';

interface Prompt {
  id: string;
  prompt: string;
}

interface Segment {
  id: number;
  text: string;
  ground_truth?: string | null;
}

interface ProcessResult {
  prompt_id: string;
  result: boolean;
  raw_response: string;
  generation_id?: string;
  cost?: number;
  usage?: any;
  success: boolean;
  error?: string;
}

class FeatureExtractor {
  private db: ReturnType<typeof drizzle>;
  private openaiClient: any;

  constructor(dbUrl: string, apiKey: string) {
    const client = postgres(dbUrl);
    this.db = drizzle(client);

    // Configure OpenAI client for OpenRouter
    this.openaiClient = openai({
      apiKey,
      baseURL: 'https://openrouter.ai/api/v1',
    });
  }

  async callOpenRouter(prompt: string): Promise<any> {
    try {
      const result = await generateText({
        model: this.openaiClient('meta-llama/llama-4-scout'),
        prompt,
        maxTokens: 10,
        temperature: 0,
      });

      return {
        content: result.text.trim().toLowerCase(),
        generation_id: result.response?.id,
        usage: result.usage,
      };
    } catch (error) {
      throw new Error(`API Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  estimateCostFromUsage(usage: any, model: string = 'meta-llama/llama-4-scout'): number | null {
    if (!usage) return null;

    // Llama 4 Scout pricing (as of 2025)
    let inputCostPerMillion = 0.08;
    let outputCostPerMillion = 0.30;

    if (model !== 'meta-llama/llama-4-scout') {
      // Default pricing for other models
      inputCostPerMillion = 0.08;
      outputCostPerMillion = 0.30;
    }

    const promptTokens = usage.promptTokens || 0;
    const completionTokens = usage.completionTokens || 0;

    const inputCost = (promptTokens / 1_000_000) * inputCostPerMillion;
    const outputCost = (completionTokens / 1_000_000) * outputCostPerMillion;

    return inputCost + outputCost;
  }

  async getGenerationCost(
    generationId: string,
    apiKey: string,
    usage: any
  ): Promise<number | null> {
    if (!generationId) {
      return this.estimateCostFromUsage(usage);
    }

    // Wait 1 second for generation data to be available
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      const response = await fetch(`https://openrouter.ai/api/v1/generation?id=${generationId}`, {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
      });

      if (response.ok) {
        const result = await response.json();
        const totalCost = result.total_cost;
        if (totalCost !== null && totalCost > 0) {
          return totalCost;
        }
      }

      return this.estimateCostFromUsage(usage);
    } catch (error) {
      return this.estimateCostFromUsage(usage);
    }
  }

  async processPrompt(
    promptData: Prompt,
    textSegment: string,
    apiKey: string,
    semaphore: { current: number; max: number }
  ): Promise<ProcessResult> {
    // Simple semaphore implementation
    while (semaphore.current >= semaphore.max) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    semaphore.current++;

    try {
      const formattedPrompt = promptData.prompt.replace('{text}', textSegment);
      const responseData = await this.callOpenRouter(formattedPrompt);
      const result = responseData.content === 'true';

      const cost = await this.getGenerationCost(
        responseData.generation_id,
        apiKey,
        responseData.usage
      );

      return {
        prompt_id: promptData.id,
        result,
        raw_response: responseData.content,
        generation_id: responseData.generation_id,
        cost,
        usage: responseData.usage,
        success: true,
      };
    } catch (error) {
      return {
        prompt_id: promptData.id,
        result: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        success: false,
        raw_response: '',
      };
    } finally {
      semaphore.current--;
    }
  }

  async runAllPrompts(
    prompts: Prompt[],
    textSegment: string,
    maxParallel: number,
    apiKey: string
  ): Promise<ProcessResult[]> {
    const semaphore = { current: 0, max: maxParallel };

    const tasks = prompts.map(prompt =>
      this.processPrompt(prompt, textSegment, apiKey, semaphore)
    );

    return Promise.all(tasks);
  }

  createFeatureVector(results: ProcessResult[]): number[] {
    // Sort results by prompt_id to ensure consistent ordering
    const sortedResults = results.sort((a, b) => a.prompt_id.localeCompare(b.prompt_id));

    // Create feature vector
    const featureVector: number[] = [];
    for (const result of sortedResults) {
      if (result.success) {
        // Convert boolean result to int (true -> 1, false -> 0)
        featureVector.push(result.result ? 1 : 0);
      } else {
        // Use 0 for failed requests
        featureVector.push(0);
      }
    }

    return featureVector;
  }

  async fetchTextSegments(modelRunId: number, limit?: number): Promise<Segment[]> {
    let query = this.db
      .select({
        id: generatedSegments.id,
        content: generatedSegments.content,
      })
      .from(generatedSegments)
      .where(eq(generatedSegments.segModelRunId, modelRunId));

    if (limit) {
      query = query.limit(limit);
    }

    const rows = await query;
    return rows.map(row => ({
      id: row.id,
      text: row.content,
      ground_truth: null,
    }));
  }

  async fetchSegmentsWithGroundTruth(limit?: number): Promise<Segment[]> {
    let query = this.db
      .select({
        id: groundTruthSegments.id,
        content: groundTruthSegments.content,
        classification: groundTruthSegments.classification,
      })
      .from(groundTruthSegments)
      .where(eq(groundTruthSegments.classification, null).not());

    if (limit) {
      query = query.limit(limit);
    }

    const rows = await query;
    return rows.map(row => ({
      id: row.id,
      text: row.content,
      ground_truth: row.classification,
    }));
  }

  async createFeatureExtractionModelRun(
    segModelRunId: number | null,
    promptsFile: string,
    maxParallel: number,
    limit?: number,
    descriptionSuffix?: string
  ): Promise<number> {
    const now = new Date();
    let name = `Feature Extraction ${now.toISOString().slice(0, 19).replace('T', ' ')}`;
    if (descriptionSuffix) {
      name += ` - ${descriptionSuffix}`;
    }

    const description = segModelRunId !== null
      ? `Feature extraction using LLM prompts from segmentation run ${segModelRunId}`
      : 'Feature extraction using LLM prompts from ground_truth_segments table (ground truth)';

    const metadata = {
      source_seg_model_run_id: segModelRunId,
      prompts_file: promptsFile,
      max_parallel: maxParallel,
      limit,
      timestamp: now.toISOString(),
      extraction_type: 'feature_extraction_llm',
      api_provider: 'OpenRouter',
      llm_model: 'meta-llama/llama-4-scout',
    };

    const result = await this.db
      .insert(modelRuns)
      .values({
        name,
        description,
        metadata: JSON.stringify(metadata),
        type: 'featureExtraction',
        createdAt: now.toISOString(),
        completed: false,
      })
      .returning({ id: modelRuns.id });

    return result[0].id;
  }

  async saveFeatureVector(
    featureExtractionModelRunId: number,
    segModelRunId: number | null,
    segmentId: number,
    featureVector: number[],
    prompts: Prompt[],
    groundTruth?: string | null,
    isGeneratedSegment: boolean = true,
    totalCost?: number | null
  ): Promise<void> {
    // Create array of prompt IDs for the prompts column
    const promptIds = prompts.sort((a, b) => a.id.localeCompare(b.id)).map(p => p.id);

    const now = new Date();
    const baseData = {
      featureExtractionModelRunId,
      segModelRunId,
      vector: featureVector,
      prompts: promptIds,
      groundTruth,
      totalCost,
      createdAt: now.toISOString(),
      updatedAt: now.toISOString(),
    };

    if (isGeneratedSegment) {
      await this.db
        .insert(featureVectors)
        .values({
          ...baseData,
          generatedSegmentId: segmentId,
        })
        .onConflictDoUpdate({
          target: [featureVectors.featureExtractionModelRunId, featureVectors.generatedSegmentId],
          set: {
            vector: featureVector,
            prompts: promptIds,
            groundTruth,
            totalCost,
            updatedAt: now.toISOString(),
          },
        });
    } else {
      await this.db
        .insert(featureVectors)
        .values({
          ...baseData,
          groundTruthSegmentId: segmentId,
        })
        .onConflictDoUpdate({
          target: [featureVectors.featureExtractionModelRunId, featureVectors.groundTruthSegmentId],
          set: {
            vector: featureVector,
            prompts: promptIds,
            groundTruth,
            totalCost,
            updatedAt: now.toISOString(),
          },
        });
    }
  }

  async processSegments(
    segModelRunId: number | null,
    promptsFile: string,
    maxParallel: number,
    apiKey: string,
    limit?: number,
    startId?: number,
    descriptionSuffix?: string,
    useSegments: boolean = false
  ): Promise<void> {
    // Load prompts
    const promptsData = await fs.readFile(promptsFile, 'utf-8');
    const prompts: Prompt[] = JSON.parse(promptsData);

    // Create new model run for this feature extraction task
    const featureExtractionModelRunId = await this.createFeatureExtractionModelRun(
      segModelRunId,
      promptsFile,
      maxParallel,
      limit,
      descriptionSuffix
    );

    console.log(`Created feature extraction model run ${featureExtractionModelRunId}`);

    // Fetch text segments based on source
    let segments: Segment[];
    if (useSegments) {
      segments = await this.fetchSegmentsWithGroundTruth(limit);
      console.log(
        `Processing ${segments.length} segments from ground_truth_segments table (ground truth data)`
      );
    } else {
      if (segModelRunId === null) {
        throw new Error('seg_model_run_id is required when not using segments table');
      }
      segments = await this.fetchTextSegments(segModelRunId, limit);
      console.log(
        `Processing ${segments.length} segments from generated_segments table (segmentation model run ${segModelRunId})`
      );
    }

    // Filter by start_id if provided
    if (startId) {
      segments = segments.filter(seg => seg.id >= startId);
    }

    // Process each segment
    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      console.log(`Processing segment ${i + 1}/${segments.length}: ID ${segment.id}`);

      // Skip if text is empty
      if (!segment.text || !segment.text.trim()) {
        console.log(`  Skipping empty segment ${segment.id}`);
        continue;
      }

      try {
        // Run feature extraction
        const results = await this.runAllPrompts(prompts, segment.text, maxParallel, apiKey);

        // Create feature vector
        const featureVector = this.createFeatureVector(results);

        // Calculate total cost for this segment
        const totalCost = results
          .filter(result => result.success)
          .reduce((sum, result) => sum + (result.cost || 0), 0);

        // Save to database
        await this.saveFeatureVector(
          featureExtractionModelRunId,
          segModelRunId,
          segment.id,
          featureVector,
          prompts,
          segment.ground_truth,
          !useSegments,
          totalCost > 0 ? totalCost : null
        );

        // Debug: Show cost calculation details
        const successfulResults = results.filter(r => r.success);
        const costs = successfulResults.map(r => r.cost || 0);
        console.log(
          `  DEBUG: ${successfulResults.length} successful prompts, individual costs: ${costs.slice(0, 5).join(', ')}... (showing first 5)`
        );
        console.log(`  DEBUG: Total cost calculated: ${totalCost}`);

        const groundTruthInfo = segment.ground_truth ? ` (ground truth: ${segment.ground_truth})` : '';
        const costInfo = totalCost > 0 ? ` (cost: $${totalCost.toFixed(6)})` : ' (cost: unknown)';
        console.log(
          `  Saved feature vector for segment ${segment.id} (shape: [${featureVector.length}])${groundTruthInfo}${costInfo}`
        );
      } catch (error) {
        console.log(`  Error processing segment ${segment.id}: ${error instanceof Error ? error.message : error}`);
        continue;
      }
    }

    // Calculate and display total cost summary
    const costSummary = await this.db
      .select({
        totalSegments: featureVectors.id.count(),
        totalCost: featureVectors.totalCost.sum(),
        avgCostPerSegment: featureVectors.totalCost.avg(),
      })
      .from(featureVectors)
      .where(eq(featureVectors.featureExtractionModelRunId, featureExtractionModelRunId));

    console.log(
      `\nProcessing complete! Feature extraction model run ID: ${featureExtractionModelRunId}`
    );
    console.log(`Processed ${costSummary[0]?.totalSegments || 0} segments`);
    if (costSummary[0]?.totalCost) {
      console.log(`Total cost: $${costSummary[0].totalCost.toFixed(6)}`);
      console.log(`Average cost per segment: $${costSummary[0].avgCostPerSegment?.toFixed(6) || 0}`);
    } else {
      console.log('Cost information not available');
    }
  }
}

async function main() {
  const { values: args } = parseArgs({
    options: {
      'seg-model-run-id': { type: 'string' },
      'prompts': { type: 'string', default: 'prompts.json' },
      'max-parallel': { type: 'string', default: '5' },
      'db-url': {
        type: 'string',
        default: 'postgresql://traindata:traindata@localhost:5433/traindata'
      },
      'limit': { type: 'string' },
      'start-id': { type: 'string' },
      'description': { type: 'string' },
      'use-segments': { type: 'boolean', default: false },
    },
  });

  // Validate arguments
  if (!args['use-segments'] && !args['seg-model-run-id']) {
    console.error('Error: --seg-model-run-id is required unless --use-segments is specified');
    process.exit(1);
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.error('Error: OPENROUTER_API_KEY environment variable not set');
    process.exit(1);
  }

  const extractor = new FeatureExtractor(args['db-url']!, apiKey);

  await extractor.processSegments(
    args['seg-model-run-id'] ? parseInt(args['seg-model-run-id']) : null,
    args.prompts!,
    parseInt(args['max-parallel']!),
    apiKey,
    args.limit ? parseInt(args.limit) : undefined,
    args['start-id'] ? parseInt(args['start-id']) : undefined,
    args.description,
    args['use-segments']!
  );
}

if (require.main === module) {
  main().catch(console.error);
}