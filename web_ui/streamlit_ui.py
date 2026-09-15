"""Streamlit frontend for the RAG Q&A API."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.pipeline.hybrid_retrieval import RETRIEVAL_CONFIG


API_BASE_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
PIPELINE_STAGE_DWELL_SECONDS = 0.32

st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --primary-color: #496a86;
    --rag-bg: var(--background-color, #ffffff);
    --rag-sidebar: var(--secondary-background-color, #f7f7f5);
    --rag-surface: color-mix(in srgb, var(--text-color, #171717) 5%, var(--background-color, #ffffff));
    --rag-surface-raised: color-mix(in srgb, var(--text-color, #171717) 2%, var(--background-color, #ffffff));
    --rag-text: var(--text-color, #171717);
    --rag-muted: color-mix(in srgb, var(--text-color, #171717) 62%, transparent);
    --rag-faint: color-mix(in srgb, var(--text-color, #171717) 42%, transparent);
    --rag-border: color-mix(in srgb, var(--text-color, #171717) 13%, transparent);
    --rag-border-strong: color-mix(in srgb, var(--text-color, #171717) 22%, transparent);
    --rag-control: var(--text-color, #262626);
    --rag-control-text: var(--background-color, #fafafa);
    --rag-hover: color-mix(in srgb, var(--text-color, #171717) 8%, var(--background-color, #ffffff));
    --accent-blue: color-mix(in srgb, #496a86 78%, var(--text-color, #171717));
    --accent-blue-soft: color-mix(in srgb, var(--accent-blue) 14%, var(--background-color, #ffffff));
    --accent-teal: color-mix(in srgb, #4f7b75 78%, var(--text-color, #171717));
    --accent-teal-soft: color-mix(in srgb, var(--accent-teal) 14%, var(--background-color, #ffffff));
    --accent-slate-soft: color-mix(in srgb, var(--accent-blue) 8%, var(--background-color, #ffffff));
    --accent-generation: var(--accent-blue);
    --accent-error: color-mix(in srgb, #9b5c5c 78%, var(--text-color, #171717));
    --accent-error-soft: color-mix(in srgb, var(--accent-error) 14%, var(--background-color, #ffffff));
    --rag-shadow: 0 1px 2px rgba(0, 0, 0, 0.08), 0 4px 14px rgba(0, 0, 0, 0.06);
}

html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--rag-bg);
    color: var(--rag-text);
}

[data-testid="stHeader"] {
    background: transparent;
}

/* The app intentionally offers only the explicit Light and Dark themes. */
[data-testid="stMainMenuItem-theme-System"] {
    display: none !important;
}

.block-container {
    max-width: 1040px;
    padding: 42px 40px 64px;
}

p, li, label, [data-testid="stMarkdownContainer"] {
    color: var(--rag-text);
}

.product-header {
    display: flex;
    align-items: center;
    gap: 11px;
    margin: 0 0 30px;
}

.product-mark {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    color: var(--accent-blue);
    border: 1px solid color-mix(in srgb, var(--accent-blue) 42%, var(--rag-border));
    border-radius: 8px;
}

.product-mark svg {
    width: 17px;
    height: 17px;
}

.product-title {
    margin: 0;
    color: var(--rag-text);
    font-size: 31px;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: -0.025em;
}

.product-subtitle {
    margin-top: 5px;
    color: var(--rag-muted);
    font-size: 14px;
    line-height: 1.5;
}

.conversation-label {
    margin: 0;
    color: var(--rag-text);
    font-size: 18px;
    font-weight: 600;
    line-height: 32px;
    letter-spacing: -0.01em;
}

.empty-state {
    padding: 34px 0 18px;
    color: var(--rag-muted);
    text-align: center;
    font-size: 15px;
}

[class*="st-key-example_questions"] {
    max-width: 820px;
    margin: 12px auto 8px;
}

[class*="st-key-example_questions"] .stButton > button {
    min-height: 38px;
    color: var(--rag-muted);
    background: var(--rag-surface-raised);
    border-color: var(--rag-border);
    font-size: 12px;
    line-height: 1.35;
    white-space: normal;
}

[class*="st-key-example_questions"] .stButton > button:hover {
    color: var(--accent-blue);
    background: var(--accent-blue-soft);
    border-color: color-mix(in srgb, var(--accent-blue) 50%, var(--rag-border));
}

.empty-state strong {
    display: block;
    margin-bottom: 8px;
    color: var(--rag-text);
    font-size: 19px;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--rag-sidebar);
    border-right: 1px solid var(--rag-border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 32px;
}

[data-testid="stSidebar"] .block-container {
    padding: 0 24px 32px;
}

.sidebar-kicker {
    margin-bottom: 4px;
    color: var(--rag-muted);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.sidebar-title {
    margin-bottom: 14px;
    color: var(--rag-text);
    font-size: 21px;
    font-weight: 600;
    letter-spacing: -0.015em;
}

/* Sidebar navigation: conversation workspace and settings are separate views. */
[class*="st-key-sidebar_view"] [data-testid="stSegmentedControl"] {
    margin-bottom: 18px;
}

[class*="st-key-sidebar_view"] [data-testid="stSegmentedControl"] button {
    min-height: 34px;
    font-size: 13px;
    font-weight: 550;
}

[class*="st-key-new_conversation"] .stButton > button {
    justify-content: flex-start;
    min-height: 42px;
    margin-bottom: 18px;
    padding: 0 12px;
    color: #ffffff !important;
    background: var(--accent-blue);
    border-color: var(--accent-blue);
    font-size: 14px;
    font-weight: 600;
}

[class*="st-key-new_conversation"] .stButton > button:hover {
    color: #ffffff !important;
    background: color-mix(in srgb, var(--accent-blue) 88%, var(--rag-text));
    border-color: color-mix(in srgb, var(--accent-blue) 88%, var(--rag-text));
}

[class*="st-key-new_conversation"] .stButton > button *,
[class*="st-key-new_conversation"] .stButton > button p {
    color: #ffffff !important;
}

.conversation-history-label {
    margin: 2px 0 8px;
    color: var(--rag-muted);
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.conversation-history-empty {
    padding: 12px 4px;
    color: var(--rag-faint);
    font-size: 13px;
}

[class*="st-key-conversation_row_"] {
    align-items: center;
    gap: 2px !important;
    margin-bottom: 2px;
    padding: 2px;
    border-radius: 8px;
}

[class*="st-key-conversation_row_"]:hover {
    background: var(--rag-hover);
}

[class*="st-key-conversation_item_"] .stButton > button {
    justify-content: flex-start;
    width: 100%;
    min-height: 36px;
    padding: 0 8px;
    overflow: hidden;
    color: var(--rag-text);
    background: transparent;
    border: 0;
    box-shadow: none;
    font-size: 13px;
    font-weight: 450;
    text-overflow: ellipsis;
    white-space: nowrap;
}

[class*="st-key-conversation_item_"] .stButton > button > div {
    justify-content: flex-start;
    width: 100%;
    min-width: 0;
    overflow: hidden;
    text-align: left;
    text-overflow: ellipsis;
}

[class*="st-key-conversation_item_"] .stButton > button:hover {
    color: var(--rag-text);
    background: transparent;
    border: 0;
}

[class*="st-key-conversation_item_"] .stButton > button[kind="primary"] {
    color: var(--accent-blue);
    font-weight: 600;
}

[class*="st-key-conversation_menu_"] {
    opacity: 0;
    transition: opacity 140ms ease;
}

[class*="st-key-conversation_row_"]:hover [class*="st-key-conversation_menu_"],
[class*="st-key-conversation_row_active_"] [class*="st-key-conversation_menu_"],
[class*="st-key-conversation_menu_"]:focus-within {
    opacity: 1;
}

[class*="st-key-conversation_menu_"] button[aria-haspopup="dialog"] {
    display: grid !important;
    place-items: center !important;
    width: 34px !important;
    min-width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 !important;
    color: var(--rag-muted) !important;
    border: 0 !important;
    background: transparent !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-size: 19px !important;
    line-height: 1 !important;
    position: relative !important;
}

[class*="st-key-conversation_menu_"] button[aria-haspopup="dialog"]:hover,
[class*="st-key-conversation_menu_"] button[aria-haspopup="dialog"][aria-expanded="true"] {
    color: var(--rag-text) !important;
    background: var(--rag-border-strong) !important;
}

[class*="st-key-conversation_menu_"] [data-testid="stPopoverButton"] > div,
[class*="st-key-conversation_menu_"] button[aria-haspopup="dialog"] > div {
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

[class*="st-key-conversation_menu_"] [data-testid="stPopoverButton"] p,
[class*="st-key-conversation_menu_"] button[aria-haspopup="dialog"] p {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    width: auto !important;
    margin: 0 !important;
    transform: translate(-50%, -54%) !important;
    text-align: center !important;
}

[class*="st-key-conversation_menu_"] svg {
    display: none !important;
}

[class*="st-key-conversation_menu_"] button[aria-haspopup="dialog"]
    [data-testid="stIconMaterial"] {
    display: none !important;
}

[data-testid="stPopoverBody"]:has([class*="st-key-conversation_popup_"]) {
    width: 168px !important;
    min-width: 168px !important;
    max-width: 168px !important;
    padding: 4px !important;
    background: color-mix(in srgb, var(--rag-text) 8%, var(--rag-sidebar));
    border: 1px solid var(--rag-border-strong);
    border-radius: 11px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
}

/* This app only uses Popover for conversation actions. Targeting the body
   directly avoids the portal wrapper overriding the compact menu width. */
div[data-testid="stPopoverBody"] {
    width: 168px !important;
    min-width: 168px !important;
    max-width: 168px !important;
    padding: 4px !important;
}

div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

[class*="st-key-conversation_popup_"] .stButton > button {
    justify-content: flex-start;
    width: 100%;
    min-height: 32px;
    padding: 0 8px;
    color: var(--rag-text);
    background: transparent;
    border: 0;
    border-radius: 7px;
    box-shadow: none;
    font-size: 15px;
    font-weight: 500;
}

[class*="st-key-conversation_popup_"] > div > [data-testid="stVerticalBlock"],
[class*="st-key-conversation_popup_"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

[data-testid="stPopoverBody"]:has([class*="st-key-conversation_popup_"])
    [data-testid="stElementContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}

[class*="st-key-conversation_popup_"] .stButton > button > div {
    justify-content: flex-start;
    width: 100%;
    gap: 9px;
    text-align: left;
}

[class*="st-key-conversation_popup_"] .stButton > button svg {
    width: 19px;
    height: 19px;
    flex: 0 0 19px;
}

[class*="st-key-conversation_popup_"] .stButton > button:hover {
    color: var(--rag-text);
    background: var(--rag-hover);
    border: 0;
}

[class*="st-key-delete_action_"] .stButton > button,
[class*="st-key-delete_action_"] .stButton > button:hover {
    color: #c84f4f;
}

[data-testid="stDialog"] [role="dialog"] {
    background: var(--rag-sidebar);
    border: 1px solid var(--rag-border-strong);
    border-radius: 14px;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
}

[data-testid="stDialog"] [data-testid="stForm"] {
    border: 0;
}

[data-testid="stDialog"] [data-testid="stTextInput"] input {
    min-height: 48px;
    color: var(--rag-text);
    background: var(--rag-surface-raised);
    border-color: var(--rag-border-strong);
    border-radius: 8px;
    font-size: 15px;
}

[data-testid="stDialog"] [data-testid="stFormSubmitButton"] button {
    min-height: 40px;
    border-radius: 8px;
}

[class*="st-key-confirm_delete_"] .stButton > button {
    color: #ffffff;
    background: #b74f4f;
    border-color: #b74f4f;
}

[class*="st-key-confirm_delete_"] .stButton > button:hover {
    color: #ffffff;
    background: #a64646;
    border-color: #a64646;
}

.sidebar-view-heading {
    margin: 2px 0 14px;
    color: var(--rag-text);
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.015em;
}

.sidebar-view-caption {
    margin: -8px 0 18px;
    color: var(--rag-muted);
    font-size: 12px;
}

.setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 0 4px;
}

.setting-label {
    color: var(--rag-text);
    font-size: 14px;
    font-weight: 500;
}

.numeric-badge {
    min-width: 28px;
    padding: 2px 8px;
    color: var(--rag-text);
    background: var(--rag-surface-raised);
    border: 1px solid var(--rag-border);
    border-radius: 7px;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    text-align: center;
}

.sidebar-spacer {
    height: 16px;
}

[data-testid="stSidebar"] [data-testid="stSlider"] {
    margin-bottom: 14px;
}

[data-testid="stSidebar"] [data-testid="stSlider"] p {
    font-size: 13px;
}

[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background-color: var(--rag-border-strong) !important;
}

[data-testid="stSidebar"] [data-testid="stToggle"] {
    margin: 3px 0 11px;
}

[data-testid="stSidebar"] [data-testid="stToggle"] label p {
    color: var(--rag-text);
    font-size: 14px;
    font-weight: 500;
}

[data-testid="stSidebar"] [role="switch"] {
    transform: scale(0.88);
    transform-origin: center right;
    transition: background-color 180ms ease, border-color 180ms ease;
}

[data-testid="stSidebar"] [role="switch"][aria-checked="false"] {
    background: var(--rag-border-strong) !important;
}

[data-testid="stSidebar"] [role="switch"][aria-checked="true"] {
    background: var(--accent-teal) !important;
}

.pipeline-card {
    margin-top: 24px;
    padding: 13px 12px 12px;
    background: var(--accent-slate-soft);
    border: 1px solid color-mix(in srgb, var(--accent-blue) 38%, var(--rag-border));
    border-left: 3px solid var(--accent-blue);
    border-radius: 11px;
    transition: border-color 160ms ease;
}

.pipeline-card:hover {
    border-color: color-mix(in srgb, var(--accent-blue) 58%, var(--rag-border));
}

.pipeline-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
}

.pipeline-label {
    color: var(--accent-blue);
}

.pipeline-label,
.system-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.system-label {
    margin-bottom: 10px;
    color: var(--rag-muted);
}

.pipeline-active-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 7px;
    color: var(--accent-teal);
    background: var(--accent-teal-soft);
    border: 1px solid color-mix(in srgb, var(--accent-teal) 38%, var(--rag-border));
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.pipeline-active-badge::before {
    content: "";
    width: 5px;
    height: 5px;
    background: var(--accent-teal);
    border-radius: 50%;
}

.pipeline-active-badge.is-error {
    color: var(--accent-error);
    background: var(--accent-error-soft);
    border-color: color-mix(in srgb, var(--accent-error) 42%, var(--rag-border));
}

.pipeline-active-badge.is-error::before {
    background: var(--accent-error);
}

.pipeline-flow {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 0;
}

.pipeline-node {
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr) auto;
    align-items: center;
    gap: 8px;
    min-height: 34px;
    padding: 5px 8px;
    color: var(--rag-text);
    background: var(--rag-surface-raised);
    border: 1px solid var(--rag-border);
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    line-height: 1.25;
    transition: border-color 160ms ease, background-color 160ms ease, opacity 160ms ease;
}

.pipeline-node.is-enabled {
    background: var(--accent-blue-soft);
    border-color: color-mix(in srgb, var(--accent-blue) 44%, var(--rag-border));
}

.pipeline-node.is-disabled {
    opacity: 0.48;
    background: transparent;
    border-style: dashed;
}

.pipeline-node.is-pending {
    background: var(--rag-surface-raised);
    border-color: var(--rag-border);
}

.pipeline-node.is-complete {
    background: var(--accent-blue-soft);
    border-color: color-mix(in srgb, var(--accent-blue) 48%, var(--rag-border));
}

.pipeline-node.is-current {
    background: var(--accent-blue-soft);
    border-color: var(--accent-blue);
    box-shadow: inset 3px 0 0 var(--accent-blue);
}

.pipeline-node.is-error {
    background: var(--accent-error-soft);
    border-color: var(--accent-error);
    box-shadow: inset 3px 0 0 var(--accent-error);
}

.pipeline-step {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    color: var(--rag-muted);
    background: var(--rag-surface-raised);
    border: 1px solid var(--rag-border);
    border-radius: 6px;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
}

.pipeline-node.is-enabled .pipeline-step {
    color: var(--accent-blue);
    border-color: color-mix(in srgb, var(--accent-blue) 48%, var(--rag-border));
}

.pipeline-node.is-current .pipeline-step {
    color: var(--accent-blue);
    border-color: var(--accent-blue);
}

.pipeline-status-dot {
    width: 6px;
    height: 6px;
    background: var(--rag-faint);
    border-radius: 50%;
}

.pipeline-node.is-complete .pipeline-status-dot {
    background: var(--accent-teal);
}

.pipeline-node.is-pending .pipeline-status-dot {
    background: transparent;
    border: 1px solid var(--accent-teal);
}

.pipeline-current-arrow {
    color: var(--accent-blue);
    font-size: 18px;
    font-weight: 700;
    line-height: 1;
}

.pipeline-error-dot {
    width: 7px;
    height: 7px;
    background: var(--accent-error);
    border-radius: 50%;
}

.pipeline-node.is-current .pipeline-current-arrow {
    animation: pipeline-pulse 1.1s ease-in-out infinite alternate;
}

@keyframes pipeline-pulse {
    from { opacity: 0.48; transform: translateX(-1px); }
    to { opacity: 1; transform: translateX(1px); }
}

.pipeline-connector {
    width: 1px;
    height: 7px;
    margin-left: 18px;
    background: var(--rag-border-strong);
}

.pipeline-legend {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px 8px;
    margin-top: 10px;
    padding-top: 9px;
    border-top: 1px solid var(--rag-border);
    color: var(--rag-muted);
    font-size: 10px;
}

.pipeline-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}

.legend-dot,
.legend-ring {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

.legend-complete { background: var(--accent-teal); }
.legend-skipped { background: var(--rag-faint); }
.legend-error { background: var(--accent-error); }
.legend-current {
    background: transparent;
    border: 1px solid var(--accent-blue);
}

.pipeline-connector.is-complete {
    background: color-mix(in srgb, var(--accent-teal) 58%, var(--rag-border));
}

.system-block {
    margin-top: 28px;
    padding-top: 20px;
    border-top: 1px solid var(--rag-border);
}

.system-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 7px 0;
    color: var(--rag-muted);
    font-size: 13px;
}

.system-dot {
    width: 4px;
    height: 4px;
    background: var(--rag-faint);
    border-radius: 50%;
}

/* Status */
.api-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin: 0 0 22px;
    color: var(--rag-muted);
    font-size: 12px;
}

.api-status-dot {
    width: 6px;
    height: 6px;
    background: var(--accent-teal);
    border-radius: 50%;
}

.api-status.is-offline .api-status-dot {
    background: #a3a3a3;
    outline: 1px solid var(--rag-border-strong);
}

[data-testid="stSidebar"] .api-status {
    margin: 0 0 24px;
    font-size: 13px;
}

/* Buttons */
.stButton > button {
    min-height: 32px;
    padding: 0 11px;
    color: var(--rag-muted);
    background: transparent;
    border: 1px solid var(--rag-border);
    border-radius: 8px;
    box-shadow: none;
    font-size: 12px;
    font-weight: 500;
    transition: color 150ms ease, background-color 150ms ease, border-color 150ms ease;
}

.stButton > button:hover {
    color: var(--rag-text);
    background: var(--rag-hover);
    border-color: var(--rag-border-strong);
}

.stButton > button:focus:not(:active) {
    color: var(--rag-text);
    border-color: var(--rag-muted);
    box-shadow: none;
}

/* Conversation */
[data-testid="stChatMessage"] {
    gap: 12px;
    margin: 18px 0;
    padding: 0;
    background: transparent;
    border: 0;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
    width: 28px;
    height: 28px;
    color: var(--rag-text);
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: 8px;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    max-width: 780px;
    font-size: 15px;
    line-height: 1.72;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 12px;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    margin: 6px 0;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    width: fit-content;
    max-width: min(82%, 760px);
    margin-left: auto;
    padding: 14px 16px;
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: 11px;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stMarkdownContainer"] p {
    margin: 0;
    font-weight: 500;
}

.citation-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 16px 0 2px;
}

.citation-tag {
    display: inline-flex;
    align-items: center;
    height: 24px;
    padding: 0 8px;
    color: var(--rag-muted) !important;
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: 7px;
    font-size: 11px;
    font-weight: 500;
    text-decoration: none !important;
    transition: color 150ms ease, background-color 150ms ease, border-color 150ms ease;
}

.citation-tag:hover {
    color: var(--accent-blue) !important;
    background: var(--accent-blue-soft);
    border-color: color-mix(in srgb, var(--accent-blue) 52%, var(--rag-border));
}

[data-testid="stExpander"] {
    margin-top: 14px;
    background: transparent;
    border: 1px solid var(--rag-border);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpanderDetails"] {
    color: var(--rag-text) !important;
    background: var(--rag-surface) !important;
}

[data-testid="stExpander"] summary {
    color: var(--rag-muted);
    font-size: 13px;
}

.source-card {
    padding: 14px 2px;
    border-bottom: 1px solid var(--rag-border);
}

.source-card:last-child {
    border-bottom: 0;
}

.source-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.source-number {
    display: inline-grid;
    place-items: center;
    width: 24px;
    height: 24px;
    color: var(--rag-text);
    background: var(--rag-surface);
    border: 1px solid var(--rag-border);
    border-radius: 7px;
    font-size: 11px;
    font-weight: 600;
}

.source-score {
    color: var(--rag-muted);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 11px;
}

.source-meta {
    margin-bottom: 7px;
    color: var(--rag-muted);
    font-size: 12px;
}

.source-text {
    color: var(--rag-text);
    font-size: 13px;
    line-height: 1.62;
}

/* Input */
[data-testid="stBottom"] {
    background: var(--rag-bg) !important;
    border-top: 0;
}

[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"] {
    background: var(--rag-bg) !important;
}

[data-testid="stBottom"] > div {
    max-width: 1040px;
    margin: 0 auto;
    padding: 16px 40px 24px;
}

[data-testid="stChatInput"] {
    position: relative;
    min-height: 64px;
    background: var(--rag-surface-raised) !important;
    border: 1px solid var(--rag-border-strong);
    border-radius: 12px;
    box-shadow: var(--rag-shadow);
    transition: border-color 150ms ease, box-shadow 150ms ease;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"] {
    background: var(--rag-surface-raised) !important;
}

[class*="st-key-chat_composer"] {
    position: relative;
    left: 50%;
    width: calc(100% + 96px);
    margin-top: 40px;
    transform: translateX(-50%);
}

[data-testid="stChatInput"]::after {
    content: "Enter 发送  ·  Shift + Enter 换行";
    position: absolute;
    right: 52px;
    bottom: 5px;
    color: var(--rag-faint);
    font-size: 10px;
    line-height: 1;
    pointer-events: none;
}

[data-testid="stChatInput"]:focus-within {
    min-height: 68px;
    border-color: var(--accent-blue);
    box-shadow: var(--rag-shadow);
}

[data-testid="stChatInput"] textarea {
    min-height: 26px !important;
    height: auto !important;
    max-height: 112px !important;
    padding-top: 1px !important;
    padding-bottom: 15px !important;
    line-height: 22px !important;
    color: var(--rag-text) !important;
    background: transparent !important;
    font-size: 15px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--rag-faint) !important;
}

[data-testid="stChatInput"] button {
    width: 34px;
    height: 34px;
    margin-right: 7px;
    color: var(--rag-control-text) !important;
    background: var(--accent-generation) !important;
    border: 0 !important;
    border-radius: 9px !important;
    transition: opacity 150ms ease, background-color 150ms ease;
}

[data-testid="stChatInput"] button:hover {
    opacity: 0.86;
}

@media (max-width: 760px) {
    .block-container {
        padding: 28px 20px 116px;
    }

    [data-testid="stBottom"] > div {
        padding: 12px 20px 18px;
    }

    .product-header {
        margin-bottom: 32px;
    }

    .product-title {
        font-size: 29px;
    }

    [class*="st-key-example_questions"] {
        margin-top: 6px;
    }

    [class*="st-key-chat_composer"] {
        left: auto;
        width: 100%;
        margin-top: 28px;
        transform: none;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        max-width: 94%;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# Streamlit's native theme menu updates its Emotion theme, but it does not expose
# those colors as CSS custom properties. Mirror the active native color-scheme to
# the variables used by this app's custom CSS so the whole interface changes.
st.html(
    """
    <script>
    (() => {
        const root = document.documentElement;

        const applyRagTheme = () => {
            const app = document.querySelector('[data-testid="stApp"]');
            if (!app) return;

            const nativeScheme = getComputedStyle(app).colorScheme.toLowerCase();
            const isDark = nativeScheme.includes('dark');
            const palette = isDark
                ? {
                    '--primary-color': '#7F9BB2',
                    '--background-color': '#101010',
                    '--secondary-background-color': '#161616',
                    '--text-color': '#F5F5F5'
                }
                : {
                    '--primary-color': '#496A86',
                    '--background-color': '#FFFFFF',
                    '--secondary-background-color': '#F7F7F5',
                    '--text-color': '#171717'
                };

            Object.entries(palette).forEach(([name, value]) => {
                root.style.setProperty(name, value);
            });
            root.dataset.ragTheme = isDark ? 'dark' : 'light';
        };

        const scheduleThemeSync = () => {
            requestAnimationFrame(applyRagTheme);
            window.setTimeout(applyRagTheme, 80);
            window.setTimeout(applyRagTheme, 240);
        };

        applyRagTheme();
        if (window.__ragThemeSyncInstalled) return;
        window.__ragThemeSyncInstalled = true;

        document.addEventListener('click', scheduleThemeSync, true);
        const observer = new MutationObserver(scheduleThemeSync);
        observer.observe(document.head, { childList: true, subtree: true });
        observer.observe(document.querySelector('[data-testid="stApp"]'), {
            attributes: true,
            attributeFilter: ['class', 'style']
        });
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

def check_api_connection() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return True, "RAG API connected"
        return False, "RAG API returned an unexpected status"
    except requests.ConnectionError:
        return False, "Cannot connect to RAG API server"
    except Exception as exc:
        return False, f"RAG API connection error: {exc}"


def api_request(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_BASE_URL}{path}", timeout=15, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else {}


def close_rename_dialog():
    st.session_state.rename_dialog_conversation_id = None


@st.dialog("Rename conversation", width="small", on_dismiss=close_rename_dialog)
def show_rename_dialog(conversation_id: str, current_title: str):
    with st.form(key=f"rename_dialog_form_{conversation_id}"):
        new_title = st.text_input(
            "Conversation name",
            value=current_title,
            max_chars=120,
            key=f"rename_dialog_value_{conversation_id}",
        )
        cancel_column, confirm_column = st.columns(2)
        cancel = cancel_column.form_submit_button("Cancel", use_container_width=True)
        confirm = confirm_column.form_submit_button(
            "Confirm", type="primary", use_container_width=True
        )
        if cancel:
            close_rename_dialog()
            st.rerun()
        if confirm:
            clean_title = new_title.strip()
            if not clean_title:
                st.error("Conversation name cannot be empty")
            else:
                api_request(
                    "PATCH",
                    f"/api/conversations/{conversation_id}",
                    json={"title": clean_title},
                )
                close_rename_dialog()
                st.rerun()


def close_delete_dialog():
    st.session_state.delete_dialog_conversation_id = None


@st.dialog("Delete conversation?", width="small", on_dismiss=close_delete_dialog)
def show_delete_dialog(conversation_id: str, conversation_title: str):
    st.markdown(
        f'This will permanently delete **{conversation_title}** and all messages in it.'
    )
    st.caption("This action cannot be undone.")
    cancel_column, delete_column = st.columns(2)
    if cancel_column.button(
        "Cancel", key=f"cancel_delete_{conversation_id}", use_container_width=True
    ):
        close_delete_dialog()
        st.rerun()
    if delete_column.button(
        "Delete",
        key=f"confirm_delete_{conversation_id}",
        type="primary",
        use_container_width=True,
    ):
        api_request("DELETE", f"/api/conversations/{conversation_id}")
        if st.session_state.get("active_conversation_id") == conversation_id:
            remaining = api_request("GET", "/api/conversations")["conversations"]
            replacement = next(
                (
                    item for item in remaining
                    if item.get("message_count", 0) == 0
                    and item.get("title") == "New conversation"
                ),
                None,
            )
            if replacement is None:
                replacement = api_request("POST", "/api/conversations", json={})
            st.session_state.pending_conversation_id = replacement["id"]
            st.session_state.loaded_conversation_id = None
        close_delete_dialog()
        st.rerun()


def stored_message_to_ui(message):
    metadata = message.get("metadata") or {}
    return {
        "role": message["role"],
        "content": message["content"],
        "retrieval": message.get("retrieval") or [],
        "citations": message.get("citations") or [],
        "lang": metadata.get("lang", "zh"),
        "use_bm25": metadata.get("use_bm25", False),
        "use_rerank": metadata.get("use_rerank", False),
    }


def format_score(result, use_bm25, use_rerank):
    if use_bm25 and use_rerank:
        return (
            f"rrf {result.get('rrf_score', 0.0):.4f} · "
            f"rerank {result.get('rerank_score', 0.0):.4f}"
        )
    if use_bm25:
        dense_score = result.get("dense_score")
        bm25_score = result.get("bm25_score")
        dense_text = f"{dense_score:.4f}" if dense_score is not None else "N/A"
        bm25_text = f"{bm25_score:.4f}" if bm25_score is not None else "N/A"
        return f"dense {dense_text} · bm25 {bm25_text} · rrf {result.get('rrf_score', 0.0):.4f}"
    return f"dense {result.get('score', 0.0):.4f}"


def render_citations(content, retrieval, message_index):
    cited = []
    for number in re.findall(r"\[(\d+)\]", content):
        index = int(number)
        if 1 <= index <= len(retrieval) and index not in cited:
            cited.append(index)
    if not cited:
        return
    tags = "".join(
        f'<a class="citation-tag" href="#source-{message_index}-{number}">[{number}]</a>'
        for number in cited
    )
    st.markdown(f'<div class="citation-row">{tags}</div>', unsafe_allow_html=True)


def render_retrieval(retrieval, use_bm25, use_rerank, message_index, expanded=False):
    if not retrieval:
        return
    with st.expander("Retrieval Results", expanded=expanded):
        cards = []
        for index, result in enumerate(retrieval, 1):
            source_values = [
                result.get("course") or result.get("subject"),
                result.get("document_title"),
                result.get("chapter_path"),
                (f"p. {result.get('page') or result.get('page_number')}"
                 if result.get("page") or result.get("page_number") else None),
                result.get("id"),
            ]
            source_meta = " · ".join(
                html.escape(str(value)) for value in source_values if value
            )
            snippet = html.escape(str(result.get("text", ""))[:260])
            score = html.escape(format_score(result, use_bm25, use_rerank))
            cards.append(
                f'<div class="source-card" id="source-{message_index}-{index}">'
                f'<div class="source-head"><span class="source-number">{index}</span>'
                f'<span class="source-score">{score}</span></div>'
                f'<div class="source-meta">{source_meta}</div>'
                f'<div class="source-text">{snippet}...</div></div>'
            )
        st.markdown("".join(cards), unsafe_allow_html=True)


def render_assistant_message(message, message_index, expanded=False):
    with st.chat_message("assistant"):
        st.markdown(message["content"])
        retrieval = message.get("retrieval", [])
        render_citations(message["content"], retrieval, message_index)
        render_retrieval(
            retrieval,
            message.get("use_bm25", False),
            message.get("use_rerank", False),
            message_index,
            expanded=expanded,
        )


def build_pipeline_html(use_bm25, use_rerank, current_stage=None, run_state="ready"):
    stages = [
        ("01", "query", "Query", True),
        ("02", "dense_retrieval", "Dense Retrieval", True),
        ("03", "bm25_retrieval", "BM25 Retrieval", use_bm25),
        ("04", "fusion", "Fusion", use_bm25),
        ("05", "bge_rerank", "BGE Rerank", use_rerank),
        ("06", "context_assembly", "Context Assembly", True),
        ("07", "llm_generation", "LLM Generation", True),
        ("08", "answer", "Answer", True),
    ]
    stage_index = {stage_id: index for index, (_, stage_id, _, _) in enumerate(stages)}
    current_index = stage_index.get(current_stage)
    parts = []

    for index, (number, stage_id, label, enabled) in enumerate(stages):
        classes = ["pipeline-node"]
        if not enabled:
            classes.append("is-disabled")
        else:
            classes.append("is-enabled")
            if current_index is None or index > current_index:
                classes.append("is-pending")
            elif stage_id == current_stage:
                classes.append("is-current")
                if run_state == "error":
                    classes.append("is-error")
            else:
                classes.append("is-complete")

        if stage_id == current_stage and enabled and run_state == "error":
            indicator = '<span class="pipeline-error-dot" aria-label="Failed stage"></span>'
        elif stage_id == current_stage and enabled:
            indicator = '<span class="pipeline-current-arrow" aria-label="Current stage">›</span>'
        else:
            indicator = '<span class="pipeline-status-dot"></span>'
        parts.append(
            f'<div class="{" ".join(classes)}" data-stage="{stage_id}">'
            f'<span class="pipeline-step">{number}</span>'
            f'<span>{label}</span>{indicator}</div>'
        )
        if index < len(stages) - 1:
            connector_state = ""
            if current_index is not None and index < current_index and enabled:
                connector_state = " is-complete"
            parts.append(f'<div class="pipeline-connector{connector_state}"></div>')

    badge = {
        "running": "Running",
        "complete": "Complete",
        "error": "Error",
    }.get(run_state, "Ready")
    badge_class = "pipeline-active-badge is-error" if run_state == "error" else "pipeline-active-badge"
    legend = (
        '<div class="pipeline-legend">'
        '<span class="pipeline-legend-item"><i class="legend-dot legend-complete"></i>Enabled / Done</span>'
        '<span class="pipeline-legend-item"><i class="legend-ring legend-current"></i>Current</span>'
        '<span class="pipeline-legend-item"><i class="legend-dot legend-skipped"></i>Skipped</span>'
        '<span class="pipeline-legend-item"><i class="legend-dot legend-error"></i>Failed</span>'
        '</div>'
    )
    return (
        '<div class="pipeline-card"><div class="pipeline-header">'
        '<div class="pipeline-label">Current Pipeline</div>'
        f'<div class="{badge_class}">{badge}</div></div>'
        f'<div class="pipeline-flow">{"".join(parts)}</div>{legend}</div>'
    )


st.markdown(
    """
    <div class="product-header">
        <div class="product-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <circle cx="12" cy="12" r="7"></circle>
                <path d="M12 5V2M12 22v-3M5 12H2M22 12h-3"></path>
            </svg>
        </div>
        <div>
            <h1 class="product-title">RAG Intelligent Q&amp;A System</h1>
            <div class="product-subtitle">Evidence-grounded retrieval &amp; generation</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "api_connected" not in st.session_state:
    connected, status_message = check_api_connection()
    st.session_state.api_connected = connected
    st.session_state.api_status_message = status_message

status_class = "" if st.session_state.api_connected else " is-offline"

if st.session_state.api_connected:
    try:
        conversation_list = api_request("GET", "/api/conversations")["conversations"]
        if not conversation_list:
            conversation_list = [api_request("POST", "/api/conversations", json={})]
        conversation_ids = [item["id"] for item in conversation_list]
        pending_conversation_id = st.session_state.pop("pending_conversation_id", None)
        if pending_conversation_id in conversation_ids:
            st.session_state.active_conversation_id = pending_conversation_id
        elif not st.session_state.get("conversation_workspace_initialized"):
            blank = next(
                (
                    item for item in conversation_list
                    if item.get("message_count", 0) == 0
                    and item.get("title") == "New conversation"
                ),
                None,
            )
            if blank is None:
                blank = api_request("POST", "/api/conversations", json={})
                conversation_list.insert(0, blank)
                conversation_ids.insert(0, blank["id"])
            st.session_state.active_conversation_id = blank["id"]
            st.session_state.conversation_workspace_initialized = True
        elif st.session_state.get("active_conversation_id") not in conversation_ids:
            st.session_state.active_conversation_id = conversation_ids[0]
    except requests.RequestException:
        conversation_list = []
        st.session_state.api_connected = False
        status_class = " is-offline"
else:
    conversation_list = []

class _NullPipelineSlot:
    def markdown(self, *args, **kwargs):
        return None


top_k = int(st.session_state.get("top_k_slider", RETRIEVAL_CONFIG["final_top_k"]))
use_bm25 = bool(st.session_state.get("use_bm25_toggle", True))
use_rerank = bool(st.session_state.get("use_rerank_toggle", True)) and use_bm25
pipeline_slot = _NullPipelineSlot()

with st.sidebar:
    sidebar_view = st.segmented_control(
        "Sidebar view",
        options=["Conversations", "Settings"],
        default="Conversations",
        key="sidebar_view",
        label_visibility="collapsed",
        width="stretch",
    )

    if sidebar_view == "Conversations":
        st.markdown('<div class="sidebar-view-heading">Conversations</div>', unsafe_allow_html=True)
        with st.container(key="new_conversation"):
            if st.button(
                "New conversation",
                icon=":material/edit_square:",
                use_container_width=True,
                key="create_conversation",
            ):
                current = next(
                    (
                        item for item in conversation_list
                        if item["id"] == st.session_state.get("active_conversation_id")
                    ),
                    None,
                )
                if (
                    current
                    and current.get("message_count", 0) == 0
                    and current.get("title") == "New conversation"
                ):
                    created = current
                else:
                    created = api_request("POST", "/api/conversations", json={})
                st.session_state.pending_conversation_id = created["id"]
                st.session_state.loaded_conversation_id = None
                st.rerun()

        st.markdown('<div class="conversation-history-label">Recent</div>', unsafe_allow_html=True)
        history_items = [
            item for item in conversation_list
            if item.get("message_count", 0) > 0 or item.get("title") != "New conversation"
        ]
        if not history_items:
            st.markdown(
                '<div class="conversation-history-empty">Your conversations will appear here.</div>',
                unsafe_allow_html=True,
            )
        for item in history_items:
            conversation_id = item["id"]
            is_active = conversation_id == st.session_state.get("active_conversation_id")
            row_key = (
                f"conversation_row_active_{conversation_id}"
                if is_active else f"conversation_row_{conversation_id}"
            )
            with st.container(key=row_key):
                title_column, menu_column = st.columns([0.86, 0.14], gap="small")
                with title_column:
                    if st.button(
                        item["title"],
                        key=f"conversation_item_{conversation_id}",
                        type="primary" if is_active else "tertiary",
                        icon=":material/chat_bubble_outline:",
                        use_container_width=True,
                    ):
                        st.session_state.active_conversation_id = conversation_id
                        st.session_state.loaded_conversation_id = None
                        st.rerun()
                with menu_column:
                    with st.popover("⋯", key=f"conversation_menu_{conversation_id}"):
                        with st.container(
                            key=f"conversation_popup_{conversation_id}", gap=None
                        ):
                            if st.button(
                                "Rename",
                                key=f"rename_action_{conversation_id}",
                                icon=":material/edit:",
                                use_container_width=True,
                            ):
                                st.session_state.delete_dialog_conversation_id = None
                                st.session_state.rename_dialog_conversation_id = conversation_id
                                st.rerun()
                            if st.button(
                                "Delete",
                                key=f"delete_action_{conversation_id}",
                                icon=":material/delete_outline:",
                                use_container_width=True,
                            ):
                                st.session_state.rename_dialog_conversation_id = None
                                st.session_state.delete_dialog_conversation_id = conversation_id
                                st.rerun()

        st.markdown(
            f'<div class="api-status{status_class}"><span class="api-status-dot"></span>'
            f'{html.escape(st.session_state.api_status_message)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="sidebar-view-heading">Settings</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-view-caption">Retrieval and generation configuration</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="api-status{status_class}"><span class="api-status-dot"></span>'
            f'{html.escape(st.session_state.api_status_message)}</div>',
            unsafe_allow_html=True,
        )
        top_k_value = st.session_state.get(
            "top_k_slider", int(RETRIEVAL_CONFIG["final_top_k"])
        )
        st.markdown(
            f'<div class="setting-row"><span class="setting-label">Max Recall Count</span>'
            f'<span class="numeric-badge">{top_k_value}</span></div>',
            unsafe_allow_html=True,
        )
        top_k = st.slider(
            "Max Recall Count",
            min_value=1,
            max_value=int(RETRIEVAL_CONFIG["max_top_k"]),
            value=int(RETRIEVAL_CONFIG["final_top_k"]),
            step=1,
            label_visibility="collapsed",
            key="top_k_slider",
        )
        use_bm25 = st.toggle(
            "Hybrid Retrieval (Dense+BM25)", value=True, key="use_bm25_toggle"
        )
        use_rerank = st.toggle(
            "Enable Reranking", value=True, disabled=not use_bm25,
            key="use_rerank_toggle",
        )
        if not use_bm25:
            use_rerank = False

        pipeline_slot = st.empty()
        pipeline_slot.markdown(
            build_pipeline_html(
                use_bm25,
                use_bm25 and use_rerank,
                st.session_state.get("pipeline_stage"),
                st.session_state.get("pipeline_run_state", "ready"),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="system-block">
                <div class="system-label">System</div>
                <div class="system-item"><span class="system-dot"></span>DeepSeek LLM</div>
                <div class="system-item"><span class="system-dot"></span>FAISS Vector Search</div>
                <div class="system-item"><span class="system-dot"></span>BM25 Retrieval</div>
                <div class="system-item"><span class="system-dot"></span>BGE Reranker</div>
                <div class="system-item"><span class="system-dot"></span>AI / ML Knowledge</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


rename_dialog_id = st.session_state.get("rename_dialog_conversation_id")
if rename_dialog_id:
    rename_dialog_item = next(
        (item for item in conversation_list if item["id"] == rename_dialog_id),
        None,
    )
    if rename_dialog_item:
        show_rename_dialog(rename_dialog_id, rename_dialog_item["title"])
    else:
        close_rename_dialog()

delete_dialog_id = st.session_state.get("delete_dialog_conversation_id")
if delete_dialog_id and not rename_dialog_id:
    delete_dialog_item = next(
        (item for item in conversation_list if item["id"] == delete_dialog_id),
        None,
    )
    if delete_dialog_item:
        show_delete_dialog(delete_dialog_id, delete_dialog_item["title"])
    else:
        close_delete_dialog()


def mark_pipeline_error():
    current_stage = st.session_state.get("pipeline_stage") or "query"
    st.session_state.pipeline_stage = current_stage
    st.session_state.pipeline_run_state = "error"
    pipeline_slot.markdown(
        build_pipeline_html(
            use_bm25, use_bm25 and use_rerank, current_stage, "error"
        ),
        unsafe_allow_html=True,
    )


active_conversation_id = st.session_state.get("active_conversation_id")
if "messages" not in st.session_state:
    st.session_state.messages = []
if (
    st.session_state.api_connected
    and active_conversation_id
    and st.session_state.get("loaded_conversation_id") != active_conversation_id
):
    try:
        stored = api_request("GET", f"/api/conversations/{active_conversation_id}/messages")
        st.session_state.messages = [stored_message_to_ui(item) for item in stored["messages"]]
        st.session_state.loaded_conversation_id = active_conversation_id
    except requests.RequestException as exc:
        st.error(f"Unable to load conversation: {exc}")

header_left, header_right = st.columns([6, 1], vertical_alignment="center")
with header_left:
    st.markdown('<div class="conversation-label">Conversation</div>', unsafe_allow_html=True)
with header_right:
    if st.button("Clear Chat", use_container_width=True):
        if active_conversation_id:
            api_request("POST", f"/api/conversations/{active_conversation_id}/clear")
        st.session_state.messages = []
        st.session_state.pipeline_stage = None
        st.session_state.pipeline_run_state = "ready"
        st.rerun()

empty_state_slot = st.empty()
if not st.session_state.messages:
    with empty_state_slot.container():
        st.markdown(
            """
            <div class="empty-state">
                <strong>Ask your knowledge base</strong>
                Answers are grounded in retrieved course materials.
            </div>
            """,
            unsafe_allow_html=True,
        )
        examples = [
            "什么是死锁的四个必要条件？",
            "TCP 为什么需要三次握手？",
            "二叉搜索树的查找过程是什么？",
        ]
        with st.container(key="example_questions"):
            example_columns = st.columns(3)
            for index, (column, question) in enumerate(zip(example_columns, examples)):
                with column:
                    if st.button(question, key=f"example_{index}", use_container_width=True):
                        st.session_state.pending_prompt = question
                        st.rerun()

for message_index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        render_assistant_message(message, message_index)

composer_slot = st.empty()
with composer_slot.container():
    with st.container(key="chat_composer"):
        submitted_prompt = st.chat_input("Ask a question about your documents...")

prompt = st.session_state.pop("pending_prompt", None) or submitted_prompt
if prompt:
    empty_state_slot.empty()
    if not st.session_state.get("api_connected"):
        st.error("RAG API server not connected, please start it first")
        st.stop()

    composer_slot.empty()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.pipeline_stage = "query"
    st.session_state.pipeline_run_state = "running"
    pipeline_slot.markdown(
        build_pipeline_html(
            use_bm25, use_bm25 and use_rerank, "query", "running"
        ),
        unsafe_allow_html=True,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/chat/stream",
                    json={
                        "question": prompt,
                        "mode": "dense",
                        "top_k": top_k,
                        "use_bm25": use_bm25,
                        "use_rerank": use_rerank,
                        "conversation_id": active_conversation_id,
                        "request_id": uuid.uuid4().hex,
                    },
                    timeout=120,
                    stream=True,
                )
                if response.status_code != 200:
                    detail = response.json().get("detail", response.text)
                    mark_pipeline_error()
                    st.error(f"API error ({response.status_code}): {detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ API error: {detail}"}
                    )
                    st.stop()

                response.encoding = "utf-8"
                answer_slot = st.empty()
                answer = ""
                retrieval = []
                answer_lang = "zh"
                stream_error = None

                for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    event = json.loads(raw_line[5:].strip())
                    event_type = event.get("type")

                    if event_type == "progress":
                        current_stage = event.get("stage")
                        st.session_state.pipeline_stage = current_stage
                        pipeline_slot.markdown(
                            build_pipeline_html(
                                use_bm25,
                                use_bm25 and use_rerank,
                                current_stage,
                                "running",
                            ),
                            unsafe_allow_html=True,
                        )
                        if current_stage in {
                            "dense_retrieval",
                            "bm25_retrieval",
                            "fusion",
                            "bge_rerank",
                            "context_assembly",
                        }:
                            # Let Streamlit flush each real backend stage to the browser.
                            time.sleep(PIPELINE_STAGE_DWELL_SECONDS)
                    elif event_type == "retrieval":
                        retrieval = event.get("retrieval", [])
                        answer_lang = event.get("lang", "zh")
                    elif event_type == "chunk":
                        answer += event.get("content", "")
                        answer_slot.markdown(answer)
                    elif event_type == "done":
                        answer = event.get("answer") or answer
                        answer_slot.markdown(answer)
                    elif event_type == "error":
                        stream_error = event.get("message", "Unknown streaming error")
                        break

                response.close()
                if stream_error:
                    mark_pipeline_error()
                    st.error(stream_error)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ {stream_error}"}
                    )
                    st.stop()
                if not answer:
                    raise RuntimeError("The API stream ended without an answer")

                st.session_state.pipeline_stage = "answer"
                st.session_state.pipeline_run_state = "complete"
                pipeline_slot.markdown(
                    build_pipeline_html(
                        use_bm25, use_bm25 and use_rerank, "answer", "complete"
                    ),
                    unsafe_allow_html=True,
                )
                new_message_index = len(st.session_state.messages)
                render_citations(answer, retrieval, new_message_index)
                render_retrieval(
                    retrieval,
                    use_bm25,
                    use_bm25 and use_rerank,
                    new_message_index,
                    expanded=True,
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "retrieval": retrieval,
                        "lang": answer_lang,
                        "use_bm25": use_bm25,
                        "use_rerank": use_bm25 and use_rerank,
                    }
                )
                st.session_state.loaded_conversation_id = None
                st.rerun()
            except requests.ConnectionError:
                mark_pipeline_error()
                st.error("Lost connection to RAG API server")
                st.session_state.api_connected = False
            except requests.Timeout:
                mark_pipeline_error()
                st.error("API request timed out (120s)")
            except Exception as exc:
                mark_pipeline_error()
                st.error(f"Error generating answer: {exc}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ Error generating answer: {exc}"}
                )
