# Prox Founding Engineer Challenge

## Project Design Document

### Vulcan OmniPro 220 Multimodal Expert Agent

---

# Vision

The goal is **not** to build another PDF chatbot.

The goal is to build an AI welding assistant that behaves like an experienced technician standing beside the user while they are setting up or troubleshooting the welder.

The assistant should:

* understand the complete manual
* understand images and diagrams
* reason across multiple sections
* retrieve precise technical values
* generate visual explanations
* create interactive tools whenever static text is insufficient

The product should feel closer to Claude Artifacts than ChatGPT.

---

# High-Level Architecture

```
                     User
                       │
                       ▼
                React Frontend
                       │
             Chat + Artifacts UI
                       │
                       ▼
               FastAPI Backend
                       │
          Claude Agent SDK Orchestrator
                       │
     ┌─────────────────┼───────────────────┐
     │                 │                   │
     ▼                 ▼                   ▼
 Manual Search     Table Lookup      Image Search
     │                 │                   │
     ▼                 ▼                   ▼
 Qdrant Vector     SQLite JSON      Image Embeddings
      DB              Store            + Metadata

                       │
                       ▼
             Artifact Generator
                       │
      SVG / React / Interactive Widgets
```

---

# Core Philosophy

The agent should never simply answer.

It should decide:

* Should I search text?
* Should I retrieve a diagram?
* Should I retrieve a table?
* Should I generate an interactive visualization?
* Should I ask for clarification?

The response should be the combination of all of these.

---

# Project Structure

```
omnipro-agent/

backend/
    app.py
    agent.py
    tools/
    retrieval/
    artifacts/
    preprocessing/

frontend/
    src/
        components/
        artifacts/
        pages/

manuals/
    pdf/

knowledge/
    chunks.json
    tables.json
    images/
    metadata.json

README.md
```

---

# Phase 1 — Manual Processing Pipeline

The manual contains multiple information types.

## 1. Text

Extract every page.

Chunk intelligently.

Store:

```
section
page
title
content
```

Embedding:

```
Qdrant
```

---

## 2. Tables

Extract every table separately.

Examples

* Duty Cycle
* Settings
* Material recommendations

Store as structured JSON.

Example

```
{
 process: MIG,
 voltage:240,
 current:200,
 duty_cycle:40
}
```

These are never retrieved using embeddings.

They are queried directly.

---

## 3. Images

Extract

* diagrams
* photos
* schematics
* wiring
* weld examples

Save every image.

```
images/

page23_img1.png
page25_img2.png
```

---

## 4. Caption Images

Run Claude Vision.

Generate

```
Page

Type

Description

Keywords
```

Example

```
{
 page:25,
 type:"Polarity Diagram",
 keywords:
[
 TIG,
 polarity,
 sockets,
 torch,
 ground
]
}
```

Now images become searchable.

---

## 5. Page Screenshots

Store screenshot of every page.

This allows manual citations.

Whenever the answer references Page 18

Show Page 18.

---

# Phase 2 — Knowledge Stores

Instead of one giant vector DB

Create specialized stores.

---

## Text Store

Qdrant

Contains

* chunks
* embeddings
* page number

---

## Table Store

SQLite

Contains

* duty cycles
* material settings
* process tables

Exact retrieval.

---

## Image Store

Qdrant

Embedding generated from image caption.

---

## Metadata Store

Maps

```
page

section

related images

related tables
```

---

# Phase 3 — Claude Agent

The Claude Agent SDK acts as the reasoning engine.

It has tools.

## Tool 1

Search Manual

Input

```
query
```

Output

Relevant chunks

---

## Tool 2

Lookup Table

Input

```
process

current

voltage
```

Output

Exact values

---

## Tool 3

Find Diagram

Input

```
query
```

Output

Relevant images

---

## Tool 4

Get Manual Page

Input

```
page
```

Output

Screenshot

---

## Tool 5

Generate Artifact

Input

```
type

data
```

Output

Interactive widget

---

# Agent Workflow

User asks

↓

Claude decides

↓

Which tools are needed?

↓

Runs tools

↓

Synthesizes information

↓

Chooses response format

↓

Returns answer

---

# Response Decision Engine

Different questions require different outputs.

## Information Question

"What is Stick welding?"

Output

* text
* source pages

---

## Numeric Question

"What is duty cycle?"

Output

* table lookup
* calculator widget

---

## Setup Question

"How do I connect TIG?"

Output

* instructions
* polarity image
* generated wiring diagram

---

## Troubleshooting Question

"My weld has porosity"

Output

* troubleshooting guide
* defect images
* decision flowchart

---

## Configuration Question

"I'm welding 3mm steel"

Output

Interactive settings configurator.

---

# Artifact System

Artifacts are reusable React components.

The backend returns

```
{
 type:"polarity_diagram",
 data:{...}
}
```

Frontend renders the correct component.

---

# Artifact 1

Duty Cycle Calculator

Inputs

Current

Voltage

Process

Outputs

Duty cycle

Safe welding duration

Cooldown duration

Slider for amperage.

---

# Artifact 2

Polarity Diagram

Generate SVG.

Example

```
Torch
↓

Negative Socket

Ground Clamp

↓

Positive Socket
```

Show cable routing visually.

---

# Artifact 3

Wire Feed Explainer

Interactive exploded diagram.

Highlight

* roller
* tensioner
* spool

User clicks components.

Description appears.

---

# Artifact 4

Troubleshooting Wizard

Example

User

"My weld has porosity."

Flow

Gas Flow?

↓

Wire Condition?

↓

Ground?

↓

Travel Speed?

↓

Diagnosis

Instead of paragraphs.

---

# Artifact 5

Settings Configurator

Inputs

Process

Material

Thickness

Voltage

Outputs

Recommended

Wire feed

Voltage

Polarity

Gas

Duty cycle

---

# Artifact 6

Duty Cycle Visualizer

Bar chart

```
████

40%
```

Explain

4 minutes weld

6 minutes cool

---

# Artifact 7

Manual Viewer

Whenever sources are cited

Display

* page screenshot
* highlighted region
* image

---

# Phase 4 — Frontend

Stack

React

Tailwind

Vite

Layout

```
----------------------------
 Chat Window

----------------------------

Artifacts

Images

Sources

Manual Pages

----------------------------
```

Responses may contain

Text

Images

Interactive widgets

SVG diagrams

Flowcharts

---

# UX Features

## Manual Citation

Every answer shows

Page number

Thumbnail

Click to enlarge.

---

## Image Preview

If discussing

Wire feed

Automatically show

Wire feed image.

---

## Clarification

User

"How do I connect it?"

Agent asks

Which process?

MIG

Flux

TIG

Stick

instead of hallucinating.

---

## Streaming

Stream Claude responses.

Artifacts appear after generation.

---

# Stretch Features

Voice mode

Speech recognition

Speech synthesis

Hands-free garage operation.

---

Camera mode

User photographs machine.

Claude identifies

Buttons

Connections

Warnings

---

Component Explorer

Clickable front panel.

Hover

Explains every control.

---

Maintenance Mode

Ask

"When should I replace the liner?"

Agent searches maintenance section.

---

Comparison Mode

Compare

MIG

vs

Flux Core

Interactive table.

---

Conversation Memory

Remember

User voltage

Material

Preferred process

Previous troubleshooting.

---

# Evaluation Checklist

The finished project should satisfy

✓ Accurate retrieval

✓ Image understanding

✓ Table understanding

✓ Cross-document reasoning

✓ Interactive visualizations

✓ Manual citations

✓ Helpful explanations

✓ Clarification questions

✓ Modern UI

✓ Easy setup

---

# Development Timeline

## Day 1

Repository

Frontend

Backend

Claude SDK

Manual extraction

Chunking

Qdrant

SQLite

---

## Day 2

Image extraction

Vision captions

Image retrieval

Agent tools

Table lookup

---

## Day 3

Artifacts

Duty calculator

Troubleshooting wizard

Polarity diagrams

Settings configurator

---

## Day 4

Frontend polish

Streaming

Manual viewer

Animations

README

Video demo

Deployment

---

# Final Goal

The final experience should feel like an intelligent digital technician rather than a chatbot.

When a user asks:

> "How do I set up TIG welding?"

the system should:

1. Understand the welding process.
2. Retrieve the relevant instructions.
3. Retrieve the correct polarity diagram.
4. Retrieve the exact manual page.
5. Generate an interactive wiring diagram.
6. Explain the setup step-by-step.
7. Cite the source page.
8. Offer a setup checklist before welding.

At every step, the agent should choose the most helpful modality—text, image, diagram, or interactive widget—rather than defaulting to plain text.
