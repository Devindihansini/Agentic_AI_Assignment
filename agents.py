# agents.py - AI Agents Implementation

import os
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config
from rag_pipeline import RAGPipeline

class SubjectClassifierAgent:
    """Agent 1: විෂය වර්ගීකරණ ඒජන්තය"""

    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.groq_api_key = Config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.llm = None
        if self.groq_api_key:
            self.llm = ChatGroq(
                model=model_name,
                temperature=0.1,
                groq_api_key=self.groq_api_key
            )

        self.prompt = ChatPromptTemplate.from_template("""
        ඔබ පහත ප්‍රශ්නය කුමන විෂයට අයත්දැයි හඳුනා ගත යුතුය.

        විෂයන්: ගණිතය, විද්‍යාව, ඉතිහාසය, භාෂාව, තාක්ෂණය, කලාව, සාමාන්‍ය

        ප්‍රශ්නය: {question}

        පිළිතුර: එක් විෂයක් පමණක් නම් කරන්න.
        විෂය:
        """)

        self.chain = self.prompt | self.llm | StrOutputParser() if self.llm else None

    def classify(self, question: str) -> str:
        """ප්‍රශ්නයේ විෂය හඳුනා ගන්න"""
        if self.llm is None:
            raise RuntimeError("Missing GROQ_API_KEY. Add it to your environment or a .env file before using the app.")
        return self.chain.invoke({"question": question}).strip()

class ExplanationAgent:
    """Agent 2: පැහැදිලි කිරීමේ ඒජන්තය"""

    def __init__(self, model_name="gpt-4o-mini"):
        self.openrouter_api_key = Config.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        self.llm = None
        if self.openrouter_api_key:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.3,
                openai_api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )

        self.rag = RAGPipeline()

        self.prompt = ChatPromptTemplate.from_template("""
        ඔබ ගණිතය, විද්‍යාව, ඉතිහාසය, භාෂාව වැනි විෂයයන් පිළිබඳ
        පැහැදිලි කිරීම් කරන ගුරුවරයෙකි.

        විෂය: {subject}
        ප්‍රශ්නය: {question}

        අදාළ තොරතුරු:
        {context}

        කරුණාකර පහත සඳහන් කරුණු සලකා බලා පිළිතුරු දෙන්න:
        1. සරල භාෂාව භාවිතා කරන්න
        2. පියවරෙන් පියවර පැහැදිලි කරන්න
        3. උදාහරණ ඇතුළත් කරන්න
        4. සිංහලෙන් පිළිතුරු දෙන්න

        පිළිතුර:
        """)

        self.chain = self.prompt | self.llm | StrOutputParser() if self.llm else None

    def explain(self, question: str, subject: str) -> str:
        """ප්‍රශ්නය පැහැදිලි කරන්න"""
        if self.llm is None:
            raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to your environment or a .env file before using the app.")
        context = self.rag.retrieve(question, subject)
        return self.chain.invoke({
            "question": question,
            "subject": subject,
            "context": context
        })

class QuizAgent:
    """Agent 3: ප්‍රශ්නෝත්තර ඒජන්තය"""

    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.groq_api_key = Config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.llm = None
        if self.groq_api_key:
            self.llm = ChatGroq(
                model=model_name,
                temperature=0.2,
                groq_api_key=self.groq_api_key
            )

        self.prompt = ChatPromptTemplate.from_template("""
        ඔබ පහත පිළිතුර මත පදනම්ව ප්‍රශ්නයක් අසන ගුරුවරයෙකි.

        මුල් ප්‍රශ්නය: {original_question}
        පැහැදිලි කිරීම: {explanation}

        කරුණාකර:
        1. ඉගෙන ගත් දේ තහවුරු කර ගැනීමට ප්‍රශ්නයක් අසන්න
        2. ප්‍රශ්නය සරල හා පැහැදිලි විය යුතුය
        3. සිංහලෙන් අසන්න

        ප්‍රශ්නය:
        """)

        self.chain = self.prompt | self.llm | StrOutputParser() if self.llm else None

    def generate_quiz(self, question: str, explanation: str) -> str:
        """ප්‍රශ්නයක් ජනනය කරන්න"""
        if self.llm is None:
            raise RuntimeError("Missing GROQ_API_KEY. Add it to your environment or a .env file before using the app.")
        return self.chain.invoke({
            "original_question": question,
            "explanation": explanation
        })

class TutorAgentSystem:
    """Main Agent System - Agents අතර සන්නිවේදනය"""

    def __init__(self):
        self.classifier = SubjectClassifierAgent()
        self.explainer = ExplanationAgent()
        self.quizzer = QuizAgent()
        self.history = []

    def process_question(self, question: str) -> str:
        """ප්‍රශ්නය සැකසීම සහ Agents අතර සන්නිවේදනය"""
        subject = self.classifier.classify(question)
        explanation = self.explainer.explain(question, subject)
        quiz = self.quizzer.generate_quiz(question, explanation)
        response = f"""
### 📚 **විෂය:** {subject}

---

### 📖 **පැහැදිලි කිරීම:**
{explanation}

---

### ❓ **දැනුම පරීක්ෂාව:**
{quiz}

---

💡 **උපදෙස:** මෙම ප්‍රශ්නයට පිළිතුරු දීමෙන් ඔබේ අවබෝධය තහවුරු කර ගන්න.
"""
        self.history.append({
            "question": question,
            "subject": subject,
            "response": response
        })
        return response

    def get_history(self):
        """සංවාද ඉතිහාසය ලබා ගන්න"""
        return self.history
