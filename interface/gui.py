# interface/gui.py
import os
import sys
import time
import json
import shutil
import threading
import urllib.request
import subprocess
from pathlib import Path
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox

from core.domain.models import RAGQuery
from core.utils.observer import Observer
from core.utils.localization.translator import Translator
from scripts.hardware_analyzer import analyze_hardware

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

RECOMMENDED_MODELS = [
    {
        "name": "Gemma 4 E4B Abliterated (Q4_K_M)",
        "url": "https://huggingface.co/OBLITERATUS/gemma-4-E4B-it-OBLITERATED/resolve/main/gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf?download=true",
        "comment": "Базовая модель (без цензуры). Требует ~6 ГБ VRAM.",
        "filename": "gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf"
    },
    {
        "name": "Gemma 4 E2B Abliterated (Q4_K_M)",
        "url": "https://huggingface.co/mradermacher/Huihui-gemma-4-E2B-it-abliterated-GGUF/resolve/main/Huihui-gemma-4-E2B-it-abliterated.Q4_K_M.gguf?download=true",
        "comment": "Легкая и быстрая модель. Идеальна для слабых ПК.",
        "filename": "Huihui-gemma-4-E2B-it-abliterated.Q4_K_M.gguf"
    }
]

class GUIInterface(ctk.CTk, Observer):
    def __init__(self, config: dict, rag_engine, logger, translator: Translator):
        super().__init__()
        self.config = config
        self.rag_engine = rag_engine
        self.logger = logger
        self.translator = translator
        self.tr = self.translator.translate
        
        self.current_query_obj = None
        self.selected_files = []
        self.citations_cache = {}
        
        self.left_panel_visible = True
        self.right_panel_visible = True
        self.stage_start_time = 0
        self.current_loaded_model_path = None
        
        self.title("Alt-RAG Studio")
        self.geometry("1400x900")
        self.minsize(1000, 600)
        
        # ПЕРЕХВАТ ЗАКРЫТИЯ ОКНА (Защита от утечки памяти)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.rag_engine.add_observer(self)
        
        self.build_ui()
        self.load_local_models()
        self.check_hardware()
        self.load_history()

    def on_closing(self):
        """Гарантированно убивает все процессы llama-server и закрывает окно."""
        self.logger.info("Закрытие приложения. Выгрузка моделей...")
        try:
            if getattr(self.rag_engine, 'synthesis_inference_engine', None):
                self.rag_engine.synthesis_inference_engine.unload_model()
            if getattr(self.rag_engine, 'retrieval_fallback_inference_engine', None):
                self.rag_engine.retrieval_fallback_inference_engine.unload_model()
        except Exception as e:
            self.logger.error(f"Ошибка при выгрузке моделей: {e}")
        
        self.destroy()
        os._exit(0) # Жесткое завершение всех потоков (закроет и консоль)

    def build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # === ЛЕВАЯ ПАНЕЛЬ (История) ===
        self.left_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)
        
        self.btn_new_chat = ctk.CTkButton(self.left_frame, text="➕ Новый чат", fg_color="#2b7a4b", hover_color="#1e5c36", command=self.start_new_chat)
        self.btn_new_chat.pack(pady=10, padx=10, fill="x")

        self.search_var = ctk.StringVar()
        self.search_var.trace("w", lambda name, index, mode: self.load_history())
        self.entry_search = ctk.CTkEntry(self.left_frame, placeholder_text="🔍 Поиск по чатам...", textvariable=self.search_var)
        self.entry_search.pack(pady=(0, 10), padx=10, fill="x")
        
        self.scroll_history = ctk.CTkScrollableFrame(self.left_frame, fg_color="transparent")
        self.scroll_history.pack(fill="both", expand=True, padx=5, pady=5)

        # === ПРАВАЯ ПАНЕЛЬ (Настройки и Модели) ===
        self.right_frame = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.right_frame.grid(row=0, column=2, sticky="nsew")
        self.right_frame.grid_propagate(False)

        self.right_tabs = ctk.CTkTabview(self.right_frame)
        self.right_tabs.pack(fill="both", expand=True, padx=10, pady=10)
        tab_settings = self.right_tabs.add(self.tr("gui_tab_settings"))
        tab_models = self.right_tabs.add(self.tr("gui_tab_models"))
        
        self.build_settings_tab(tab_settings)
        self.build_models_tab(tab_models)

        # === ЦЕНТРАЛЬНАЯ ПАНЕЛЬ (Чат) ===
        self.center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.center_frame.grid_rowconfigure(1, weight=1)
        self.center_frame.grid_columnconfigure(1, weight=1)

        self.top_bar = ctk.CTkFrame(self.center_frame, height=40, fg_color="transparent")
        self.top_bar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        
        self.btn_toggle_left = ctk.CTkButton(self.top_bar, text="☰", width=30, command=self.toggle_left)
        self.btn_toggle_left.pack(side="left")
        
        self.lbl_chunking = ctk.CTkLabel(self.top_bar, text=self.tr("gui_ind_chunking"), text_color="gray", font=("Arial", 14, "bold"))
        self.lbl_chunking.pack(side="left", padx=15)
        self.lbl_retrieval = ctk.CTkLabel(self.top_bar, text=self.tr("gui_ind_retrieval"), text_color="gray", font=("Arial", 14, "bold"))
        self.lbl_retrieval.pack(side="left", padx=15)
        self.lbl_synthesis = ctk.CTkLabel(self.top_bar, text=self.tr("gui_ind_synthesis"), text_color="gray", font=("Arial", 14, "bold"))
        self.lbl_synthesis.pack(side="left", padx=15)
        
        self.btn_toggle_right = ctk.CTkButton(self.top_bar, text="⚙", width=30, command=self.toggle_right)
        self.btn_toggle_right.pack(side="right")

        self.chat_box = ctk.CTkTextbox(self.center_frame, wrap="word", font=("Consolas", 14))
        self.chat_box.grid(row=1, column=0, columnspan=3, sticky="nsew")
        self.chat_box._textbox.tag_config("citation", foreground="#00BFFF", underline=True)
        
        self.progress_frame = ctk.CTkFrame(self.center_frame, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        self.progress_frame.grid_remove()
        
        self.lbl_progress_status = ctk.CTkLabel(self.progress_frame, text="Статус...", font=("Arial", 12))
        self.lbl_progress_status.pack(side="top", anchor="w")
        self.lbl_found_chunks = ctk.CTkLabel(self.progress_frame, text="Найдено фрагментов: 0", font=("Arial", 12, "bold"), text_color="#00BFFF")
        self.lbl_found_chunks.pack(side="top", anchor="w")
        self.progressbar = ctk.CTkProgressBar(self.progress_frame)
        self.progressbar.pack(side="top", fill="x", pady=2)
        self.progressbar.set(0)

        self.input_frame = ctk.CTkFrame(self.center_frame)
        self.input_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        self.input_frame.grid_columnconfigure(1, weight=1)

        self.btn_attach = ctk.CTkButton(self.input_frame, text=self.tr("gui_btn_attach"), width=60, command=self.select_files)
        self.btn_attach.grid(row=0, column=0, padx=5, pady=10)

        self.entry_query = ctk.CTkEntry(self.input_frame, placeholder_text=self.tr("gui_placeholder_query"))
        self.entry_query.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        
        self.entry_query.bind("<Control-v>", self.paste_text)
        self.entry_query.bind("<Return>", lambda e: self.start_rag_process())

        self.btn_send = ctk.CTkButton(self.input_frame, text=self.tr("gui_btn_send"), width=80, command=self.start_rag_process)
        self.btn_send.grid(row=0, column=2, padx=5, pady=10)

        self.btn_early_stop = ctk.CTkButton(self.input_frame, text=self.tr("gui_btn_stop"), width=120, fg_color="#b53b3b", hover_color="#8a2a2a", command=self.trigger_early_stop)
        self.btn_early_stop.grid(row=0, column=3, padx=5, pady=10)
        self.btn_early_stop.configure(state="disabled")

        self.lbl_file_count = ctk.CTkLabel(self.center_frame, text=self.tr("gui_files_attached").format(count=0), text_color="gray")
        self.lbl_file_count.grid(row=4, column=0, columnspan=3, sticky="w", padx=5)

    def paste_text(self, event):
        try:
            text = self.clipboard_get()
            self.entry_query.insert("insert", text)
        except Exception:
            pass
        return "break"

    def start_new_chat(self):
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")
        self.selected_files = []
        self.lbl_file_count.configure(text=self.tr("gui_files_attached").format(count=0))
        self.current_query_obj = None

    def toggle_left(self):
        if self.left_panel_visible:
            self.left_frame.grid_remove()
        else:
            self.left_frame.grid()
        self.left_panel_visible = not self.left_panel_visible

    def toggle_right(self):
        if self.right_panel_visible:
            self.right_frame.grid_remove()
        else:
            self.right_frame.grid()
        self.right_panel_visible = not self.right_panel_visible

    def build_settings_tab(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll, text=self.tr("gui_device")).pack(anchor="w", pady=(5, 0))
        self.device_var = ctk.StringVar(value=self.config.get("inference", {}).get("device_type", "auto"))
        ctk.CTkOptionMenu(scroll, values=["auto", "cuda12", "cuda13", "vulkan", "cpu_x64"], variable=self.device_var).pack(fill="x", pady=5)
        
        self.lbl_hw_rec = ctk.CTkLabel(scroll, text="...", text_color="green", font=("Arial", 11))
        self.lbl_hw_rec.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(scroll, text=self.tr("gui_language")).pack(anchor="w")
        self.lang_var = ctk.StringVar(value=self.config.get("language", "ru"))
        ctk.CTkOptionMenu(scroll, values=["ru", "en"], variable=self.lang_var, command=self.change_language).pack(fill="x", pady=5)

        self.lbl_temp = ctk.CTkLabel(scroll, text=f"{self.tr('gui_temp')} 0.70")
        self.lbl_temp.pack(anchor="w", pady=(10,0))
        self.slider_temp = ctk.CTkSlider(scroll, from_=0.0, to=1.0, number_of_steps=100, command=self.update_temp_label)
        self.slider_temp.set(0.7)
        self.slider_temp.pack(fill="x", pady=5)

        ctk.CTkLabel(scroll, text=self.tr("gui_strategy")).pack(anchor="w", pady=(10,0))
        self.strategy_var = ctk.StringVar(value="1")
        ctk.CTkOptionMenu(scroll, values=["1 (BestWindow)", "2 (Keywords)", "3 (Window)"], variable=self.strategy_var).pack(fill="x", pady=5)
        
        ctk.CTkLabel(scroll, text=self.tr("gui_keywords")).pack(anchor="w")
        self.entry_keywords = ctk.CTkEntry(scroll, placeholder_text="Например: антуанетта, цвет")
        self.entry_keywords.pack(fill="x", pady=5)

        ctk.CTkLabel(scroll, text=self.tr("gui_prompt_retrieval")).pack(anchor="w", pady=(10,0))
        self.txt_prompt_retrieval = ctk.CTkTextbox(scroll, height=120)
        self.txt_prompt_retrieval.pack(fill="x", pady=5)
        
        ret_prompt = "Фрагмент:\n{chunk_content} Основываясь только на следующем фрагменте текста, представленном выше, ответь, можно ли с помощью текста выше дать ответ на вопрос '{prompt}'. Строго запрещено давать пояснения. Ответ надо дать строго одним словом: 'да' или 'нет'.\n\n \n\nОтвет:"
        self.txt_prompt_retrieval.insert("1.0", ret_prompt)

        ctk.CTkLabel(scroll, text=self.tr("gui_prompt_synthesis")).pack(anchor="w", pady=(10,0))
        self.txt_prompt_synthesis = ctk.CTkTextbox(scroll, height=120)
        self.txt_prompt_synthesis.pack(fill="x", pady=5)
        
        syn_prompt = "Используя следующий контекст:\n{context}\n\nОтветь на вопрос: {question}\n\nПожалуйста, после каждого утверждения, которое основывается на контексте, укажи точную цитату из контекста в формате [ЦИТАТА: \"...\"]. Если информация не найдена в контексте, ответь, что информация отсутствует."
        self.txt_prompt_synthesis.insert("1.0", syn_prompt)

        ctk.CTkButton(scroll, text=self.tr("gui_btn_update"), fg_color="#454545", hover_color="#5a5a5a", command=self.update_engine).pack(fill="x", pady=20)

    def update_temp_label(self, val):
        self.lbl_temp.configure(text=f"{self.tr('gui_temp')} {val:.2f}")

    def change_language(self, new_lang: str):
        self.config["language"] = new_lang
        config_path = Path("configs") / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("Инфо", self.tr("gui_msg_lang_changed"))
        except Exception as e:
            pass

    def build_models_tab(self, parent):
        self.scroll_models = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.scroll_models.pack(fill="both", expand=True)

        ctk.CTkLabel(self.scroll_models, text=self.tr("gui_installed_models"), font=("Arial", 14, "bold")).pack(anchor="w", pady=(5,5))
        self.installed_models_frame = ctk.CTkFrame(self.scroll_models, fg_color="transparent")
        self.installed_models_frame.pack(fill="x", pady=5)
        
        self.model_var = ctk.StringVar()
        
        ctk.CTkLabel(self.scroll_models, text=self.tr("gui_recommended_models"), font=("Arial", 14, "bold")).pack(anchor="w", pady=(20, 5))
        self.rec_models_frame = ctk.CTkFrame(self.scroll_models, fg_color="transparent")
        self.rec_models_frame.pack(fill="x", expand=True)
        self.draw_recommended_models()

    def draw_installed_models_list(self):
        for widget in self.installed_models_frame.winfo_children():
            widget.destroy()
        
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        models = [f.name for f in model_dir.glob("*.gguf")]
        
        if models:
            self.opt_models = ctk.CTkOptionMenu(self.installed_models_frame, variable=self.model_var, values=models)
            self.opt_models.pack(fill="x", pady=(0, 5))
            self.model_var.set(models[0])
            
            # КНОПКА ТЕПЕРЬ ПОД ВЫПАДАЮЩИМ СПИСКОМ НА ВСЮ ШИРИНУ
            btn_del = ctk.CTkButton(self.installed_models_frame, text="🗑 Удалить модель", fg_color="#b53b3b", hover_color="#8a2a2a", command=self.delete_current_model)
            btn_del.pack(fill="x")
        else:
            self.opt_models = ctk.CTkOptionMenu(self.installed_models_frame, variable=self.model_var, values=[self.tr("gui_no_models")])
            self.opt_models.pack(fill="x")

    def delete_current_model(self):
        m = self.model_var.get()
        if m and m != self.tr("gui_no_models"):
            if messagebox.askyesno("Удаление", f"Точно удалить модель {m}?"):
                try:
                    os.remove(Path("models") / m)
                    self.load_local_models()
                    messagebox.showinfo("Успех", "Модель успешно удалена.")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось удалить: {e}")

    def draw_recommended_models(self):
        for widget in self.rec_models_frame.winfo_children():
            widget.destroy()
            
        for mod in RECOMMENDED_MODELS:
            frame = ctk.CTkFrame(self.rec_models_frame, fg_color="#2b2b2b")
            frame.pack(fill="x", pady=5)
            ctk.CTkLabel(frame, text=mod["name"], font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5,0))
            ctk.CTkLabel(frame, text=mod["comment"], font=("Arial", 11), text_color="gray", wraplength=280, justify="left").pack(anchor="w", padx=10)
            
            action_frame = ctk.CTkFrame(frame, fg_color="transparent")
            action_frame.pack(fill="x", padx=10, pady=5)
            
            if (Path("models") / mod["filename"]).exists():
                ctk.CTkLabel(action_frame, text="Установлено ✓", text_color="green").pack(side="right")
            else:
                btn = ctk.CTkButton(action_frame, text=self.tr("gui_btn_download"), width=80, height=24)
                btn.configure(command=lambda m=mod, f=action_frame, b=btn: self.download_model_in_frame(m, f, b))
                btn.pack(side="right")

    def download_model_in_frame(self, mod_info, parent_frame, btn_widget):
        btn_widget.destroy()
        url = mod_info["url"]
        target_path = Path("models") / mod_info["filename"]
        
        pbar = ctk.CTkProgressBar(parent_frame)
        pbar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        pbar.set(0)
        lbl_pct = ctk.CTkLabel(parent_frame, text="0%")
        lbl_pct.pack(side="right")

        def download_thread():
            try:
                def reporthook(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = (block_num * block_size) / total_size
                        self.after(0, lambda: pbar.set(min(1.0, percent)))
                        self.after(0, lambda: lbl_pct.configure(text=f"{int(percent*100)}%"))
                urllib.request.urlretrieve(url, target_path, reporthook)
                self.after(0, self.load_local_models)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка скачивания: {e}"))
                self.after(0, self.draw_recommended_models)

        threading.Thread(target=download_thread, daemon=True).start()

    def check_hardware(self):
        hw = analyze_hardware()
        self.lbl_hw_rec.configure(text=self.tr("gui_hw_rec").format(rec=hw['recommendation'], vram=hw['vram_gb']))
        
    def load_local_models(self):
        self.draw_installed_models_list()
        self.draw_recommended_models()

    def load_history(self):
        for w in self.scroll_history.winfo_children():
            w.destroy()
            
        out_dir = Path("user_data/outputs")
        if not out_dir.exists():
            ctk.CTkLabel(self.scroll_history, text=self.tr("gui_history_empty")).pack(pady=20)
            return

        sessions = sorted([d for d in out_dir.iterdir() if d.is_dir()], reverse=True)
        search_q = self.search_var.get().lower()

        count = 0
        for session in sessions:
            try:
                dt = datetime.strptime(session.name, "%y%m%d_%H%M%S")
                date_str = dt.strftime("%d.%m %H:%M")
            except ValueError:
                date_str = session.name

            question = "Без вопроса"
            q_file = session / "query" / "query.json"
            if q_file.exists():
                try:
                    with open(q_file, 'r', encoding='utf-8') as f:
                        q_text = json.load(f).get("question", "Без вопроса")
                        question = q_text[:35] + "..." if len(q_text) > 35 else q_text
                except: pass

            if search_q and search_q not in question.lower() and search_q not in date_str.lower():
                continue

            count += 1
            row_frame = ctk.CTkFrame(self.scroll_history, fg_color="#333333")
            row_frame.pack(fill="x", pady=2)
            row_frame.grid_columnconfigure(0, weight=1)
            
            btn_load = ctk.CTkButton(row_frame, text=f"[{date_str}]\n{question}", fg_color="transparent", anchor="w", command=lambda p=session: self.show_history_session(p))
            btn_load.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            
            btn_del = ctk.CTkButton(row_frame, text="✖", width=30, fg_color="#8a2a2a", hover_color="#b53b3b", command=lambda p=session: self.delete_history_session(p))
            btn_del.grid(row=0, column=1, padx=2, pady=2)

        if count == 0:
            ctk.CTkLabel(self.scroll_history, text="Ничего не найдено").pack(pady=20)

    def delete_history_session(self, session_path: Path):
        if messagebox.askyesno("Удаление", "Удалить этот чат навсегда?"):
            try:
                shutil.rmtree(session_path)
                self.load_history()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось удалить: {e}")

    def show_history_session(self, session_path: Path):
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        
        query_file = session_path / "query" / "query.json"
        ans_file = session_path / "answer" / "final_answer.json"
        
        if query_file.exists():
            with open(query_file, 'r', encoding='utf-8') as f:
                self.log_to_chat(json.load(f).get("question", ""), is_user=True)
                
        if ans_file.exists():
            with open(ans_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                class DummyResult: pass
                res = DummyResult()
                res.answer = data.get("answer", "")
                self.display_final_answer(res)
        else:
            self.chat_box.insert("end", "\n[Ответ не найден или процесс был прерван]\n")
            
        self.chat_box.configure(state="disabled")

    def select_files(self):
        files = filedialog.askopenfilenames(title="Выберите файлы", filetypes=[("All Files", "*.*")])
        if len(files) > 10:
            messagebox.showwarning("Внимание", self.tr("gui_msg_files_limit"))
        
        self.selected_files = [Path(f) for f in files]
        self.lbl_file_count.configure(text=self.tr("gui_files_attached").format(count=len(self.selected_files)))
        self.log_to_chat(f"[{datetime.now().strftime('%H:%M:%S')}] Прикреплено {len(self.selected_files)} файлов.")

    def log_to_chat(self, message: str, is_user=False):
        self.chat_box.configure(state="normal")
        if is_user:
            self.chat_box.insert("end", f"\nВы: {message}\n\n", "user")
        else:
            self.chat_box.insert("end", f"{message}\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def display_final_answer(self, result):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", "\n=== ОТВЕТ ===\n")
        
        import re
        parts = re.split(r'(\[ЦИТАТА:?[^\]]*\])', result.answer)
        
        for part in parts:
            if part.startswith("[ЦИТАТА"):
                cit_id = f"cit_{len(self.citations_cache)}"
                self.citations_cache[cit_id] = part
                
                start_index = self.chat_box.index("insert")
                self.chat_box.insert("end", " [Источн.] ")
                end_index = self.chat_box.index("insert")
                
                self.chat_box._textbox.tag_add(cit_id, start_index, end_index)
                self.chat_box._textbox.tag_config(cit_id, foreground="#00BFFF", underline=True)
                self.chat_box._textbox.tag_bind(cit_id, "<Button-1>", lambda e, cid=cit_id: self.on_citation_click(cid))
            else:
                self.chat_box.insert("end", part)
                
        self.chat_box.insert("end", "\n================\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def on_citation_click(self, cit_id):
        text = self.citations_cache.get(cit_id, "Текст не найден")
        popup = ctk.CTkToplevel(self)
        popup.title(self.tr("gui_source_viewer"))
        popup.geometry("600x400")
        txt = ctk.CTkTextbox(popup, wrap="word", font=("Arial", 14))
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", self.tr("gui_source_text").format(text=text))
        txt.configure(state="disabled")

    def trigger_early_stop(self):
        if self.current_query_obj:
            self.current_query_obj.early_stop = True
            self.log_to_chat(self.tr("gui_sys_early_stop"))
            self.btn_early_stop.configure(state="disabled")

    def update_pipeline_ui(self, stage):
        act = "#00BFFF"
        inact = "gray"
        self.lbl_chunking.configure(text_color=act if stage == "chunking" else inact)
        self.lbl_retrieval.configure(text_color=act if stage == "retrieval" else inact)
        self.lbl_synthesis.configure(text_color=act if stage == "synthesis" else inact)

    def start_rag_process(self):
        query_text = self.entry_query.get().strip()
        if not query_text or not self.selected_files:
            messagebox.showwarning("Внимание", self.tr("gui_msg_no_query"))
            return

        model_name = self.model_var.get()
        if not model_name or model_name == self.tr("gui_no_models"):
            messagebox.showerror("Ошибка", "Выберите установленную модель.")
            return

        new_model_path = Path("models") / model_name
        
        if self.current_loaded_model_path != new_model_path:
            self.rag_engine.synthesis_inference_engine.unload_model()
            if getattr(self.rag_engine, 'retrieval_fallback_inference_engine', None):
                self.rag_engine.retrieval_fallback_inference_engine.unload_model()
            self.current_loaded_model_path = new_model_path

        self.rag_engine.config.synthesis.synthesis_prompt = self.txt_prompt_synthesis.get("1.0", "end-1c")
        self.rag_engine.config.retrieval.retriever_prompt = self.txt_prompt_retrieval.get("1.0", "end-1c")
        self.rag_engine.config.synthesis_inference.temperature = self.slider_temp.get()
        
        strat_val = int(self.strategy_var.get().split()[0])
        self.rag_engine.config.retrieval.strategy_type = strat_val
        if strat_val == 2:
            self.rag_engine.config.retrieval.keywords = [k.strip() for k in self.entry_keywords.get().split(",") if k.strip()]
        
        self.rag_engine.config.synthesis_inference.model_path = new_model_path
        self.rag_engine.config.retrieval_inference.model_path = new_model_path

        self.log_to_chat(query_text, is_user=True)
        self.entry_query.delete(0, "end")
        self.btn_send.configure(state="disabled")
        self.btn_early_stop.configure(state="normal")
        self.progress_frame.grid()
        self.progressbar.set(0)
        self.lbl_found_chunks.configure(text="Найдено фрагментов: 0")
        
        output_dir = Path("user_data/outputs") / datetime.now().strftime("%y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.current_query_obj = RAGQuery(question=query_text, input_path=self.selected_files[0].parent, output_dir=output_dir)
        
        threading.Thread(target=self._run_engine_thread, daemon=True).start()

    def _run_engine_thread(self):
        try:
            result = self.rag_engine.run(self.current_query_obj)
            self.after(0, self.display_final_answer, result)
            self.after(0, self.load_history)
        except Exception as e:
            self.after(0, self.log_to_chat, f"\n[ОШИБКА] {str(e)}\n")
        finally:
            self.after(0, self._reset_ui_state)

    def _reset_ui_state(self):
        self.btn_send.configure(state="normal")
        self.btn_early_stop.configure(state="disabled")
        self.update_pipeline_ui("none")
        self.progress_frame.grid_remove()
        self.current_query_obj = None

    def update(self, message_type: str, data: dict):
        def update_gui():
            stage = data.get("stage", "none")
            
            if message_type == "status":
                msg = data.get("message", "")
                self.lbl_progress_status.configure(text=msg)
                self.stage_start_time = time.time()
                
                if "Чанкинг" in msg or "chunking" in stage: self.update_pipeline_ui("chunking")
                elif "Поиск" in msg or "retrieval" in msg or "retrieval" in stage: self.update_pipeline_ui("retrieval")
                elif "Синтез" in msg or "synthesis" in msg or "synthesis" in stage: self.update_pipeline_ui("synthesis")
                
            elif message_type == "progress":
                current = data.get("current", 0)
                total = data.get("total", 1)
                
                if total > 0:
                    pct = current / total
                    self.progressbar.set(pct)
                    
                    elapsed = time.time() - self.stage_start_time
                    if current > 0:
                        eta_seconds = (elapsed / current) * (total - current)
                        eta_str = time.strftime('%M:%S', time.gmtime(eta_seconds))
                        self.lbl_progress_status.configure(text=f"Выполнено: {int(pct*100)}% | Осталось: ~{eta_str}")

                if stage == "retrieval":
                    self.update_pipeline_ui("retrieval")
            
            elif message_type == "complete":
                if stage == "retrieval":
                    count = data.get("relevant_chunks_count", 0)
                    self.lbl_found_chunks.configure(text=f"Найдено фрагментов: {count}")
                    self.log_to_chat(f"[Ретривер] Найдено релевантных фрагментов: {count}")

        self.after(0, update_gui)

    def update_engine(self):
        self.log_to_chat(self.tr("gui_sys_update_start"))
        bat_content = """@echo off
chcp 65001 > nul
title Alt-RAG Updater Worker
echo Подготовка к обновлению... Закрытие программы...
timeout /t 2 /nobreak > nul
taskkill /F /IM python.exe > nul 2>&1
taskkill /F /IM llama-server.exe > nul 2>&1
call .\\rag_venv\\Scripts\\activate.bat
python scripts\\update_llamacpp.py
echo Запуск интерфейса...
start run_gui.bat
del "%~f0"
"""
        with open("temp_updater.bat", "w", encoding="utf-8") as f:
            f.write(bat_content)
        subprocess.Popen("temp_updater.bat", creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.on_closing() # Гарантированно все выгружаем и закрываем

    def run(self):
        self.mainloop()