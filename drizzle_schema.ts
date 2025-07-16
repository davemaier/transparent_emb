import {
  pgTable,
  index,
  serial,
  integer,
  text,
  timestamp,
  foreignKey,
  unique,
  doublePrecision,
  jsonb,
  boolean,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const sentences = pgTable(
  "sentences",
  {
    id: serial().primaryKey().notNull(),
    articleId: integer("article_id").notNull(),
    articleTitle: text("article_title").notNull(),
    text: text().notNull(),
    originalLabel: text("original_label").notNull(),
    currentLabel: text("current_label").notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
    updatedAt: timestamp("updated_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_sentences_article_id").using(
      "btree",
      table.articleId.asc().nullsLast().op("int4_ops"),
    ),
  ],
);

export const modelPredictions = pgTable(
  "model_predictions",
  {
    id: serial().primaryKey().notNull(),
    sentenceId: integer("sentence_id").notNull(),
    modelRunId: integer("model_run_id").notNull(),
    prediction: text().notNull(),
    confidence: doublePrecision(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_model_predictions_model_run_id").using(
      "btree",
      table.modelRunId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_model_predictions_sentence_id").using(
      "btree",
      table.sentenceId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.sentenceId],
      foreignColumns: [sentences.id],
      name: "model_predictions_sentence_id_fkey",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.modelRunId],
      foreignColumns: [modelRuns.id],
      name: "model_predictions_model_run_id_fkey",
    }).onDelete("cascade"),
    unique("model_predictions_sentence_id_model_run_id_key").on(
      table.sentenceId,
      table.modelRunId,
    ),
  ],
);

export const modelRuns = pgTable("model_runs", {
  id: serial().primaryKey().notNull(),
  name: text().notNull(),
  description: text(),
  metadata: jsonb(),
  type: text(),
  createdAt: timestamp("created_at", {
    withTimezone: true,
    mode: "string",
  }).default(sql`CURRENT_TIMESTAMP`),
});

export const migrations = pgTable("migrations", {
  id: serial().primaryKey().notNull(),
  name: text().notNull(),
  appliedAt: timestamp("applied_at", {
    withTimezone: true,
    mode: "string",
  }).default(sql`CURRENT_TIMESTAMP`),
});

export const textTable = pgTable("text", {
  id: serial().primaryKey().notNull(),
  content: text().notNull(),
  createdAt: timestamp("created_at", {
    withTimezone: true,
    mode: "string",
  }).default(sql`CURRENT_TIMESTAMP`),
});

export const segments = pgTable(
  "segments",
  {
    id: serial().primaryKey().notNull(),
    name: text().notNull(),
    content: text().notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
    textId: integer("text_id").notNull(),
    groundTruth: text("ground_truth"),
    reason: text("reason"),
  },
  (table) => [
    index("idx_segments_text_id").using(
      "btree",
      table.textId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.textId],
      foreignColumns: [textTable.id],
      name: "segments_text_id_fkey",
    }).onDelete("cascade"),
  ],
);

export const embeddings = pgTable(
  "embeddings",
  {
    id: serial().primaryKey().notNull(),
    segmentId: integer("segment_id").notNull(),
    modelRunId: integer("model_run_id").notNull(),
    embedding: doublePrecision().array().notNull(),
    modelName: text("model_name").notNull(),
    embeddingDimension: integer("embedding_dimension").notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_embeddings_segment_id").using(
      "btree",
      table.segmentId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_embeddings_model_run_id").using(
      "btree",
      table.modelRunId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.segmentId],
      foreignColumns: [segments.id],
      name: "embeddings_segment_id_fkey",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.modelRunId],
      foreignColumns: [modelRuns.id],
      name: "embeddings_model_run_id_fkey",
    }).onDelete("cascade"),
    unique("embeddings_segment_id_model_run_id_key").on(
      table.segmentId,
      table.modelRunId,
    ),
  ],
);

export const generatedSegments = pgTable(
  "generated_segments",
  {
    id: serial().primaryKey().notNull(),
    segModelRunId: integer("seg_model_run_id").notNull(),
    textId: integer("text_id").notNull(),
    name: text(),
    content: text().notNull(),
    reason: text(),
    confidence: doublePrecision(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_generated_segments_seg_model_run_id").using(
      "btree",
      table.segModelRunId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_generated_segments_text_id").using(
      "btree",
      table.textId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.segModelRunId],
      foreignColumns: [modelRuns.id],
      name: "generated_segments_seg_model_run_id_fkey",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.textId],
      foreignColumns: [textTable.id],
      name: "generated_segments_text_id_fkey",
    }).onDelete("cascade"),
  ],
);

export const segmentPredictions = pgTable(
  "segment_predictions",
  {
    id: serial().primaryKey().notNull(),
    generatedSegmentId: integer("generated_segment_id").notNull(),
    modelRunId: integer("model_run_id").notNull(),
    prediction: text().notNull(),
    confidence: doublePrecision(),
    metadata: jsonb(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_segment_predictions_generated_segment_id").using(
      "btree",
      table.generatedSegmentId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_segment_predictions_model_run_id").using(
      "btree",
      table.modelRunId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.generatedSegmentId],
      foreignColumns: [generatedSegments.id],
      name: "segment_predictions_generated_segment_id_fkey",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.modelRunId],
      foreignColumns: [modelRuns.id],
      name: "segment_predictions_model_run_id_fkey",
    }).onDelete("cascade"),
    unique("segment_predictions_generated_segment_id_model_run_id_key").on(
      table.generatedSegmentId,
      table.modelRunId,
    ),
  ],
);

export const tempNonSegmentedFragments = pgTable(
  "temp_non_segmented_fragments",
  {
    id: text().primaryKey().notNull(),
    textId: integer("text_id").notNull(),
    content: text().notNull(),
    startPosition: integer("start_position"),
    endPosition: integer("end_position"),
    metadata: jsonb(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
    updatedAt: timestamp("updated_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_temp_non_segmented_fragments_text_id").using(
      "btree",
      table.textId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.textId],
      foreignColumns: [textTable.id],
      name: "temp_non_segmented_fragments_text_id_fkey",
    }).onDelete("cascade"),
  ],
);

export const segmentFeatures = pgTable(
  "segment_features",
  {
    id: serial().primaryKey().notNull(),
    generatedSegmentId: integer("generated_segment_id").notNull(),
    modelRunId: integer("model_run_id").notNull(),

    // Climate relatedness
    isClimateRelated: boolean("is_climate_related").notNull().default(false),

    // Threat parameters (6 features)
    threatWerWas: boolean("threat_wer_was").notNull().default(false),
    threatWann: boolean("threat_wann").notNull().default(false),
    threatWo: boolean("threat_wo").notNull().default(false),
    threatWarum: boolean("threat_warum").notNull().default(false),
    threatWie: boolean("threat_wie").notNull().default(false),
    threatWodurch: boolean("threat_wodurch").notNull().default(false),

    // Solution parameters (6 features)
    solutionWer: boolean("solution_wer").notNull().default(false),
    solutionWann: boolean("solution_wann").notNull().default(false),
    solutionWo: boolean("solution_wo").notNull().default(false),
    solutionWarum: boolean("solution_warum").notNull().default(false),
    solutionWieWodurch: boolean("solution_wie_wodurch")
      .notNull()
      .default(false),
    solutionVonWas: boolean("solution_von_was").notNull().default(false),

    // Final classification results
    isThreat: boolean("is_threat").notNull().default(false),
    isSolution: boolean("is_solution").notNull().default(false),
    finalLabel: text("final_label").notNull().default("Neutral"),
    confidence: doublePrecision().notNull().default(0.0),

    // Raw LLM responses for debugging
    rawResponses: jsonb("raw_responses").notNull().default({}),

    // Timestamps
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
    updatedAt: timestamp("updated_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_segment_features_generated_segment_id").using(
      "btree",
      table.generatedSegmentId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_segment_features_model_run_id").using(
      "btree",
      table.modelRunId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_segment_features_is_climate_related").using(
      "btree",
      table.isClimateRelated.asc().nullsLast(),
    ),
    index("idx_segment_features_final_label").using(
      "btree",
      table.finalLabel.asc().nullsLast(),
    ),
    index("idx_segment_features_created_at").using(
      "btree",
      table.createdAt.asc().nullsLast(),
    ),
    index("idx_segment_features_segment_model").using(
      "btree",
      table.generatedSegmentId.asc().nullsLast().op("int4_ops"),
      table.modelRunId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.generatedSegmentId],
      foreignColumns: [generatedSegments.id],
      name: "segment_features_generated_segment_id_fkey",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.modelRunId],
      foreignColumns: [modelRuns.id],
      name: "segment_features_model_run_id_fkey",
    }).onDelete("cascade"),
    unique("segment_features_generated_segment_id_model_run_id_key").on(
      table.generatedSegmentId,
      table.modelRunId,
    ),
  ],
);

export const featureVectors = pgTable(
  "feature_vectors",
  {
    id: serial().primaryKey().notNull(),
    featureExtractionModelRunId: integer("feature_extraction_model_run_id").notNull(),
    segModelRunId: integer("seg_model_run_id"),
    generatedSegmentId: integer("generated_segment_id"),
    groundTruthSegmentId: integer("ground_truth_segment_id"),
    featureVector: jsonb("feature_vector").notNull(),
    promptIds: jsonb("prompt_ids").notNull(),
    groundTruth: text("ground_truth"),
    totalCost: doublePrecision("total_cost"),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
    updatedAt: timestamp("updated_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_feature_vectors_seg_model_run_id").using(
      "btree",
      table.segModelRunId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_feature_vectors_feature_extraction_model_run_id").using(
      "btree",
      table.featureExtractionModelRunId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_feature_vectors_generated_segment_id").using(
      "btree",
      table.generatedSegmentId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_feature_vectors_ground_truth_segment_id").using(
      "btree",
      table.groundTruthSegmentId.asc().nullsLast().op("int4_ops"),
    ),
    index("idx_feature_vectors_created_at").using(
      "btree",
      table.createdAt.asc().nullsLast(),
    ),
    foreignKey({
      columns: [table.featureExtractionModelRunId],
      foreignColumns: [modelRuns.id],
      name: "feature_vectors_feature_extraction_model_run_id_fkey",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.segModelRunId],
      foreignColumns: [modelRuns.id],
      name: "feature_vectors_seg_model_run_id_fkey",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.generatedSegmentId],
      foreignColumns: [generatedSegments.id],
      name: "fk_feature_vectors_generated_segment",
    }).onDelete("cascade"),
    foreignKey({
      columns: [table.groundTruthSegmentId],
      foreignColumns: [segments.id],
      name: "fk_feature_vectors_ground_truth_segment",
    }).onDelete("cascade"),
    unique("unique_feature_extraction_generated_segment").on(
      table.featureExtractionModelRunId,
      table.generatedSegmentId,
    ),
    unique("unique_feature_extraction_ground_truth_segment").on(
      table.featureExtractionModelRunId,
      table.groundTruthSegmentId,
    ),
  ],
);