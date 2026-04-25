"""
llm_engine.py
Wrapper for Groq API to provide financial insights and chart explanations.
"""

import streamlit as st
from groq import Groq

def get_groq_client(api_key: str):
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

def get_llm_explanation(prompt: str, api_key: str, system_prompt: str = "You are a professional financial advisor and portfolio manager. Provide concise, expert-level explanations."):
    client = get_groq_client(api_key)
    if not client:
        return "⚠️ Groq API key is missing or invalid. Please check the sidebar."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.3,
            max_tokens=500,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling Groq API: {str(e)}"

def explain_chart(chart_name: str, data_summary: str, api_key: str):
    prompt = f"""
    Analyze the following '{chart_name}' visualization for a multi-asset portfolio.
    
    Data Summary:
    {data_summary}
    
    Provide 2-3 brief, punchy bullet points on what this chart reveals about the portfolio's health, risk, or return.
    """
    return get_llm_explanation(prompt, api_key)

def explain_tax_logic(return_pct: float, risk_profile: str, api_key: str):
    prompt = f"""
    Explain how the tax-adjusted return is calculated for a '{risk_profile}' portfolio with an expected pre-tax return of {return_pct:.2%}.
    Mention Indian tax contexts like LTCG for equity (12.5% above 1.25L) and debt (flat 30% or slab rate).
    Explain why tax-adjustment is critical for realistic planning.
    Keep it under 100 words.
    """
    return get_llm_explanation(prompt, api_key)

def explain_monte_carlo(n_simulations: int, top_sharpe: float, api_key: str):
    prompt = f"""
    We ran {n_simulations} Monte Carlo simulations. The best Sharpe ratio found was {top_sharpe:.2f}.
    Briefly explain why this method is better than just picking the highest returning asset.
    """
    return get_llm_explanation(prompt, api_key)

def get_portfolio_summary(weights_dict: dict, stats: dict, api_key: str):
    prompt = f"""
    Provide a professional summary of the following portfolio:
    Allocations: {weights_dict}
    Stats: Return: {stats['return']:.2%}, Volatility: {stats['volatility']:.2%}, Sharpe: {stats['sharpe']:.2f}
    
    Highlight the core strength of this allocation in 2-3 sentences.
    """
    return get_llm_explanation(prompt, api_key)

def get_final_recommendation(risk_profile: str, horizon: int, api_key: str):
    prompt = f"""
    The user has a '{risk_profile}' risk profile and a {horizon}-year investment horizon.
    Provide a concise, punchy final recommendation on how they should approach this portfolio (e.g., rebalancing frequency, stay the course).
    Max 50 words.
    """
    return get_llm_explanation(prompt, api_key)

def get_chat_response(messages: list, context: str, api_key: str):
    """
    Handle interactive Q&A with the portfolio context.
    'messages' is a list of {'role': 'user/assistant', 'content': '...'}
    """
    client = get_groq_client(api_key)
    if not client:
        return "⚠️ Groq API key is missing or invalid."

    system_prompt = f"""
    You are an expert AI Financial Advisor. 
    A user has just completed a portfolio optimization process. 
    Here is the context of their optimized portfolio:
    {context}
    
    Answer the user's questions based on this data. Be professional, data-driven, and clear.
    If they ask for general advice, relate it back to their specific risk profile and horizon.
    """
    
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        chat_completion = client.chat.completions.create(
            messages=full_messages,
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=800,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"
