# interface/webui.py
import os
import sys

# Жестко прописываем путь к корню, чтобы Python видел папки core и scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import shutil
import urllib.request
import subprocess
import threading
import gradio as gr
from pathlib import Path
from datetime import datetime

from core.domain.models import RAGQuery
from scripts.hardware_analyzer import analyze_hardware

def get_rag_engine():
    from main import initialize_system
    from core.utils.config_loader import load_rag_config
    from core.factories.engine_factory import create_rag_engine
    
    BASE_DIR = Path(__file__).parent.parent
    config, logger, translator = initialize_system(BASE_DIR / 'configs' / 'config.json')
    rag_config = load_rag_config(BASE_DIR / 'configs' / 'rag_engine_config.json')
    engine = create_rag_engine(rag_config, logger, translator)
    return engine, config

rag_engine, sys_config = get_rag_engine()

status_store = {"msg": "Ожидание...", "progress": 0.0, "found": 0, "is_done": False, "result": None}

class GradioObserver:
    def __init__(self):
        self.start_t = time.time()
        
    def update(self, msg_type, data):
        global status_store
        stage = data.get("stage", "")
        
        if msg_type == "status":
            status_store["msg"] = data.get("message", "")
            self.start_t = time.time()
            
        elif msg_type == "progress":
            c, t = data.get("current", 0), data.get("total", 1)
            if t > 0:
                pct = c / t
                status_store["progress"] = pct
                elapsed = time.time() - self.start_t
                if c > 0:
                    eta = (elapsed / c) * (t - c)
                    eta_str = time.strftime('%M:%S', time.gmtime(eta))
                    status_store["msg"] = f"Обработка [{stage}] {int(pct*100)}%. Осталось: ~{eta_str}"
                    
        elif msg_type == "complete":
            if stage == "retrieval":
                status_store["found"] = data.get("relevant_chunks_count", 0)

obs = GradioObserver()
rag_engine.add_observer(obs)

CUSTOM_CSS = """
body {
    background: linear-gradient(-45deg, #0f172a, #16213e, #1a1a2e, #0f3460);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #e2e8f0;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.gradio-container {
    background-color: transparent !important;
}
.gr-box, .gr-panel, .gr-form {
    background-color: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
#component-0 {
    max-width: 1400px;
    margin: auto;
}
"""

def scan_history():
    out_dir = Path("user_data/outputs")
    if not out_dir.exists(): return []
    sessions = sorted([d for d in out_dir.iterdir() if d.is_dir()], reverse=True)
    choices = []
    for s in sessions:
        try:
            dt = datetime.strptime(s.name, "%y%m%d_%H%M%S").strftime("%d.%m %H:%M")
        except: dt = s.name
        q_text = "Без вопроса"
        q_file = s / "query" / "query.json"
        if q_file.exists():
            try:
                with open(q_file, 'r', encoding='utf-8') as f:
                    q_text = json.load(f).get("question", "")[:30] + "..."
            except: pass
        choices.append((f"[{dt}] {q_text}", str(s)))
    return choices

def load_history_chat(session_path):
    sess = Path(session_path)
    q_file = sess / "query" / "query.json"
    ans_file = sess / "answer" / "final_answer.json"
    chat = []
    if q_file.exists():
        with open(q_file, 'r', encoding='utf-8') as f:
            chat.append((json.load(f).get("question", ""), None))
    if ans_file.exists():
        with open(ans_file, 'r', encoding='utf-8') as f:
            chat[-1] = (chat[-1][0], json.load(f).get("answer", ""))
    return chat

def get_models():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    return [f.name for f in model_dir.glob("*.gguf")]

def delete_model(m_name):
    if m_name:
        try:
            os.remove(Path("models") / m_name)
            return gr.update(choices=get_models(), value=None), "Удалено"
        except Exception as e:
            return gr.update(), f"Ошибка: {e}"
    return gr.update(), ""

def update_engine_script():
    bat_content = """@echo off
chcp 65001 > nul
timeout /t 2 /nobreak > nul
taskkill /F /IM python.exe > nul 2>&1
taskkill /F /IM llama-server.exe > nul 2>&1
call .\\rag_venv\\Scripts\\activate.bat
python scripts\\update_llamacpp.py
start run_webui.bat
del "%~f0"
"""
    with open("temp_updater.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)
    subprocess.Popen("temp_updater.bat", creationflags=subprocess.CREATE_NEW_CONSOLE)
    sys.exit(0)

def run_rag_stream(query, files, model_name, temp, p_ret, p_syn, strat):
    global status_store
    status_store = {"msg": "Запуск движка...", "progress": 0.0, "found": 0, "is_done": False, "result": None}
    
    if not query or not files:
        yield "Ошибка: Введите запрос и загрузите файлы", [("Ошибка", "Запрос или файлы пусты.")], 0
        return
    if not model_name:
        yield "Ошибка: Выберите модель", [("Ошибка", "Модель не выбрана.")], 0
        return

    temp_dir = Path("user_data/temp_uploads")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    for f in files:
        shutil.copy(f.name, temp_dir / Path(f.name).name)

    new_model_path = Path("models") / model_name
    if getattr(rag_engine, "current_loaded_model_path", None) != new_model_path:
        rag_engine.synthesis_inference_engine.unload_model()
        if getattr(rag_engine, 'retrieval_fallback_inference_engine', None):
            rag_engine.retrieval_fallback_inference_engine.unload_model()
        rag_engine.current_loaded_model_path = new_model_path

    rag_engine.config.synthesis_inference.model_path = new_model_path
    rag_engine.config.retrieval_inference.model_path = new_model_path
    rag_engine.config.synthesis.synthesis_prompt = p_syn
    rag_engine.config.retrieval.retriever_prompt = p_ret
    rag_engine.config.synthesis_inference.temperature = temp
    rag_engine.config.retrieval.strategy_type = int(strat[0])

    out_dir = Path("user_data/outputs") / datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    
    q_obj = RAGQuery(question=query, input_path=temp_dir, output_dir=out_dir)
    
    def worker():
        try:
            res = rag_engine.run(q_obj)
            status_store["result"] = res.answer
        except Exception as e:
            status_store["result"] = f"ОШИБКА: {str(e)}"
        status_store["is_done"] = True

    t = threading.Thread(target=worker)
    t.start()

    while not status_store["is_done"]:
        time.sleep(0.5)
        stat = f"{status_store['msg']} | Найдено чанков: {status_store['found']}"
        yield stat, [(query, "Генерация ответа...")], status_store["progress"]
        
    yield "Готово!", [(query, status_store["result"])], 1.0


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🧠 Alt-RAG Studio Web")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("История"):
                    hist_dropdown = gr.Dropdown(choices=scan_history(), label="Сохраненные чаты")
                    btn_load_hist = gr.Button("Загрузить чат")
                    btn_new_chat = gr.Button("➕ Новый чат", variant="primary")
                    
                with gr.Tab("Настройки"):
                    hw = analyze_hardware()
                    gr.Markdown(f"**Рекомендовано:** {hw['recommendation']} ({hw['vram_gb']}GB VRAM)")
                    temp_sl = gr.Slider(0.0, 1.0, 0.7, step=0.01, label="Температура")
                    strat_dd = gr.Dropdown(["1 (BestWindow)", "2 (Keywords)", "3 (Window)"], value="1 (BestWindow)", label="Стратегия")
                    
                    p_ret = gr.Textbox(label="Промпт Ретривера", lines=3, value="Фрагмент:\n{chunk_content} Основываясь только на следующем фрагменте текста, представленном выше, ответь, можно ли с помощью текста выше дать ответ на вопрос '{prompt}'. Строго запрещено давать пояснения. Ответ надо дать строго одним словом: 'да' или 'нет'.\n\n \n\nОтвет:")
                    p_syn = gr.Textbox(label="Промпт Синтеза", lines=4, value="Используя следующий контекст:\n{context}\n\nОтветь на вопрос: {question}\n\nПожалуйста, после каждого утверждения, которое основывается на контексте, укажи точную цитату из контекста в формате [ЦИТАТА: \"...\"]. Если информация не найдена в контексте, ответь, что информация отсутствует.")
                    btn_upd = gr.Button("Обновить llama.cpp")
                    
                with gr.Tab("Модели"):
                    mod_dd = gr.Dropdown(choices=get_models(), label="Установленные модели")
                    btn_del_mod = gr.Button("Удалить модель", variant="stop")
                    del_status = gr.Markdown("")
                    
                    gr.Markdown("**Рекомендуемые:**")
                    RECOMMENDED_MODELS = [
                        {"name": "Gemma 4 E4B (Q4)", "url": "https://huggingface.co/OBLITERATUS/gemma-4-E4B-it-OBLITERATED/resolve/main/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf?download=true", "filename": "gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf", "comment": "Базовая"},
                        {"name": "Gemma 4 E2B (Q4)", "url": "https://huggingface.co/mradermacher/Huihui-gemma-4-E2B-it-abliterated-GGUF/resolve/main/Huihui-gemma-4-E2B-it-abliterated.Q4_K_M.gguf?download=true", "filename": "Huihui-gemma-4-E2B-it-abliterated.Q4_K_M.gguf", "comment": "Легкая"}
                    ]
                    for m in RECOMMENDED_MODELS:
                        gr.Markdown(f"- **{m['name']}**\n  {m['comment']}")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Диалог")
            status_md = gr.Markdown("Ожидание...")
            prog_bar = gr.Slider(0.0, 1.0, 0.0, interactive=False, label="Прогресс")
            
            with gr.Row():
                files_up = gr.File(file_count="multiple", label="Перетащите файлы сюда (Drag & Drop)")
            
            with gr.Row():
                query_in = gr.Textbox(label="Запрос", placeholder="Введите запрос и нажмите Отправить...", scale=4)
                btn_send = gr.Button("Отправить", variant="primary", scale=1)

    btn_send.click(
        run_rag_stream, 
        inputs=[query_in, files_up, mod_dd, temp_sl, p_ret, p_syn, strat_dd], 
        outputs=[status_md, chatbot, prog_bar]
    )
    btn_load_hist.click(load_history_chat, inputs=[hist_dropdown], outputs=[chatbot])
    btn_new_chat.click(lambda: [], outputs=[chatbot])
    btn_del_mod.click(delete_model, inputs=[mod_dd], outputs=[mod_dd, del_status])
    btn_upd.click(update_engine_script)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)