---
title: "Compiler Notes"
excerpt_separator: "<!--more-->"
classes: "wide"
categories:
  - Notes

tags:
  - 
  - 
---


## Compiler Overview

A **compiler** is a program that takes other programs and prepares them for execution.

<p align="center">
  <img src="/images/cse110/high_level_compilers.png" width = "80%">
</p>

At a high level we have three main components, a front-end which is responsible for converting the source language into an intermediate representation (IR), an optimizer which optimizes the IR code, and a back-end which is reponsible for dealing with the target language. 

There are two fundamental principles in compiler design:  
1. The compiler must preserve the meaning of the input program.  
2. The compiler must discernibly improve the input program.

The back end maps the optimized IR program onto the bounded resources of the target machine’s ISA in a way that makes efficient use of resources. Since the back-end process IR code, it assumes that all the code generated from the optimizer is free of syntactic or semantic error. 
<p align="center">
  <img src="/images/cse110/compilers_2.png" width = "80%">
</p>


### Front-end

<p align="center">
  <img src="/images/cse110/front_end.png" width = "40%">
</p>

The front end works to understand the source program and record the results of that analysis in IR form. At a high level, it first checks the syntax and semantics of the source input code, and if its valid, it creates an IR representation of the code. If the input code is invalid, it will return a diagnostic error to the user. 

The parser fits the words from the scanner to a rule-based model of the input language’s syntax, called a **grammar**.
When checking the syntax of a program, the parser compares the program's structure to the formal definition of a language's syntax. The parser does this by first matching a program's input to distinct tokens, it then tries to match the stream of tokens into the grammar following its **production rules**.

Its final responsibility is to generate an IR of the code. An **IR** is a representation of a code plus other data structure(s) that contain additional information. This can be done as a graph or tree structure.


### Optimizer 

The optimizer tries to improve the IR so that it produces a more efficient execution. This can involve rewriting the IR that the front-end creates to make it more efficient. This can be done in either a monolithic single pass, which is the most efficient, or multi-passes, which is less complex. The optimizer is an IR-to-IR transformer.  

Making code more efficient can have multiple meanings, either by reducing the time it takes for a program to run, or by decreasing the file size of a program. 

One technique that an optimizer uses is a loop invariance, which can rewrite a loop to reduce the amount of computation required. 

<p align="center">
  <img src="/images/cse110/loop_invariance.png" width = "60%">
</p>

Most optimizations consist of an analysis and a transformation. The analysis determines where the compiler can safely and profitably apply the transformation.
  - *Data-flow analysis* reasons, at compile time, about the flow of values at runtime.
  - *Dependence analysis* uses number-theoretic tests to reason about the rela- tive independence of memory references. This is used to disambiguate array-element references. 


### Back-end

The backend is responsible for converting the IR code handed to it by either the optimizer or front-end and emitting code for the target machine. It is responsible for solving three primary problems:
  -  *instruction selection*: converting IR operations into the target program's ISA  
  - *instruction scheduling*: selecting an order for operations  
  - *register allocation*: deciding where each value should reside in each given register at each point in  the code




## Parsing 

### Expressing Syntax
The task of the parser is to determine whether or not some stream of words fits into the syntax of the parser’s intended source language.

While regular expressions are powerful and provide a concise notation for describing language, it lacks the power to describe most programming languages. Instead we often use CFGs. A **context-free grammar** is a set of rules, or *productions*, that describe how to form sentences. 

Formally, a context-free grammar $G$ is a quadruple $$(T, NT, S, P)$$ where:

- $$T$$ is the set of terminal symbols, or words, in the language $$L(G)$$. Terminal symbols correspond to syntactic categories returned by the scanner.   

- $$NT$$ is the set of nonterminal symbols. They are syntactic variables introduced to provide abstraction and structure in the productions of $$G$$.  

- $$S$$ is a nonterminal symbol designated as the start symbol or goal symbol of
the grammar. $$S$$ represents the set of sentences in $L(G)$.  

- $$P$$ is the set of productions or rewrite rules in $G$. Each rule in $P$ has the form $NT → (T ∪ NT )+$ ; that is, it replaces a single nonterminal with a string of one or more grammar symbols.

To begin a **derivation** (a sequence of rewriting steps where we start with the grammar's start symbol and end with a sentence in the language), we follow the process of 
  - pick a nonterminal symbol, $\alpha$, in the prototype string
  - choose a grammar rule, $\alpha \to \beta$
  - rewrite $\alpha$ with $\beta$

The **sentential form** is a string of any valid derivation which contains both terminal and nonterminal tokens.
