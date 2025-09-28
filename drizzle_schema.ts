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
  json,
  boolean,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const modelRuns = pgTable("model_runs", {
  id: serial().primaryKey().notNull(),
  completed: boolean(),
  name: text().notNull(),
  description: text(),
  metadata: json(),
  type: text().notNull(),
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
  prod: boolean(),
});

export const groundTruthSegments = pgTable(
  "ground_truth_segments",
  {
    id: serial().primaryKey().notNull(),
    name: text().notNull(),
    content: text().notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
    textId: integer("text_id").notNull(),
    classification: text("classification"),
    metadata: jsonb("metadata"),
    start: integer("start"),
    length: integer("length"),
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

export const groundTruthEmbeddings = pgTable(
  "ground_truth_embeddings",
  {
    id: serial().primaryKey().notNull(),
    segmentId: integer("segment_id").notNull(),
    modelRunId: integer("model_run_id").notNull(),
    embedding: doublePrecision().array().notNull(),
    classification: text(),
    embeddingDimension: integer("embedding_dimension").notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
    metadata: jsonb(),
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
      foreignColumns: [groundTruthSegments.id],
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
    start: integer("start"),
    length: integer("length"),
    content: text().notNull(),
    metadata: jsonb("metadata"),
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
    classification: text("classification").notNull(),
    confidence: doublePrecision(),
    metadata: jsonb(),
    reason: text().notNull(),
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

export const nonSegmentedFragments = pgTable(
  "non_segmented_fragments",
  {
    id: serial().primaryKey().notNull(),
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
    }),
    processed: timestamp("processed_at", {
      withTimezone: true,
      mode: "string",
    }).default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_non_segmented_fragments_text_id").using(
      "btree",
      table.textId.asc().nullsLast().op("int4_ops"),
    ),
    foreignKey({
      columns: [table.textId],
      foreignColumns: [textTable.id],
      name: "non_segmented_fragments_text_id_fkey",
    }).onDelete("cascade"),
  ],
);

export const featureVectors = pgTable(
  "feature_vectors",
  {
    id: serial().primaryKey().notNull(),
    featureExtractionModelRunId: integer(
      "feature_extraction_model_run_id",
    ).notNull(),
    segModelRunId: integer("seg_model_run_id"),
    generatedSegmentId: integer("generated_segment_id"),
    groundTruthSegmentId: integer("ground_truth_segment_id"),
    vector: integer("vector").array().notNull(),
    prompts: text("prompts").array().notNull(),
    groundTruth: text("ground_truth"),
    totalCost: doublePrecision("total_cost"),
    metadata: jsonb("metadata"),
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
      foreignColumns: [groundTruthSegments.id],
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
