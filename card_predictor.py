# card_predictor.py

import re
import logging
import time
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
import pytz

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BENIN_TZ = pytz.timezone("Africa/Porto-Novo")

SUIT_TO_VALUE_MAP = {
    "♠️": "Q", "♦️": "K", "♣️": "J", "❤️": "A", "♥️": "A"
}

SYMBOL_MAP = {0: '✅0️⃣', 1: '✅1️⃣', 2: '✅2️⃣', 'lost': '❌'}

class CardPredictor:
    def __init__(self, telegram_message_sender=None):
        self.HARDCODED_SOURCE_ID = -1003424179389
        self.HARDCODED_PREDICTION_ID = -1003362820311
        
        self.predictions = self._load_data('predictions.json')
        self.processed_messages = self._load_data('processed.json', is_set=True)
        self.last_prediction_time = self._load_data('last_prediction_time.json', is_scalar=True) or 0
        self.last_predicted_game_number = self._load_data('last_predicted_game_number.json', is_scalar=True) or 0
        self.consecutive_fails = self._load_data('consecutive_fails.json', is_scalar=True) or 0
        self.pending_edits = self._load_data('pending_edits.json')
        
        raw_config = self._load_data('channels_config.json')
        self.config_data = raw_config if isinstance(raw_config, dict) else {}
        self.target_channel_id = self.config_data.get('target_channel_id') or self.HARDCODED_SOURCE_ID
        self.prediction_channel_id = self.config_data.get('prediction_channel_id') or self.HARDCODED_PREDICTION_ID
        
        self.telegram_message_sender = telegram_message_sender
        self.sequential_history = self._load_data('sequential_history.json')
        self.inter_data = self._load_data('inter_data.json')
        self.is_inter_mode_active = True
        self.smart_rules = self._load_data('smart_rules.json')
        self.last_analysis_time = self._load_data('last_analysis_time.json', is_scalar=True) or 0
        self.collected_games = self._load_data('collected_games.json', is_set=True)
        self.last_report_sent = self._load_data('last_report_sent.json')
        self.last_inter_update_time = self._load_data('last_inter_update.json', is_scalar=True) or 0
        self.quarantined_rules = self._load_data('quarantined_rules.json')
        
        self.prediction_cooldown = 30

    def _load_data(self, filename, is_set=False, is_scalar=False):
        try:
            if not os.path.exists(filename):
                return set() if is_set else (None if is_scalar else {})
            with open(filename, 'r') as f:
                content = f.read().strip()
                if not content: return set() if is_set else (None if is_scalar else {})
                data = json.loads(content)
                if is_set: return set(data)
                if isinstance(data, dict): return {int(k) if k.isdigit() else k: v for k, v in data.items()}
                return data
        except: return set() if is_set else (None if is_scalar else {})

    def _save_data(self, data, filename):
        try:
            if isinstance(data, set): data = list(data)
            with open(filename, 'w') as f: json.dump(data, f, indent=4)
        except: pass

    def _save_all_data(self):
        self._save_data(self.predictions, 'predictions.json')
        self._save_data(self.processed_messages, 'processed.json')
        self._save_data(self.last_prediction_time, 'last_prediction_time.json')
        self._save_data(self.last_predicted_game_number, 'last_predicted_game_number.json')
        self._save_data(self.consecutive_fails, 'consecutive_fails.json')
        self._save_data(self.inter_data, 'inter_data.json')
        self._save_data(self.sequential_history, 'sequential_history.json')
        self._save_data(self.smart_rules, 'smart_rules.json')
        self._save_data(self.collected_games, 'collected_games.json')
        self._save_data(self.last_report_sent, 'last_report_sent.json')
        self._save_data(self.last_inter_update_time, 'last_inter_update.json')
        self._save_data(self.quarantined_rules, 'quarantined_rules.json')

    def now(self): return datetime.now(BENIN_TZ)
    def is_in_session(self): return True

    def get_inter_version(self):
        if not self.last_inter_update_time: return "Base neuve"
        return datetime.fromtimestamp(self.last_inter_update_time, BENIN_TZ).strftime("%Y-%m-%d | %Hh%M")

    def extract_game_number(self, message):
        match = re.search(r'#N(\d+)\.', message, re.IGNORECASE) or re.search(r'🔵(\d+)🔵', message)
        return int(match.group(1)) if match else None

    def extract_card_details(self, content):
        normalized = content.replace("♥️", "❤️")
        return re.findall(r'(\d+|[AKQJ])(♠️|❤️|♦️|♣️)', normalized, re.IGNORECASE)

    def get_first_two_cards_info(self, message):
        match = re.search(r'\(([^)]*)\)', message)
        if not match: return []
        details = self.extract_card_details(match.group(1))
        return [f"{v.upper()}{'♥️' if c == '❤️' else c}" for v, c in details[:2]]

    def get_all_cards_in_first_group(self, message):
        match = re.search(r'\(([^)]*)\)', message)
        if not match: return []
        details = self.extract_card_details(match.group(1))
        return [f"{v.upper()}{'♥️' if c == '❤️' else c}" for v, c in details]

    def is_final_result_structurally_valid(self, text):
        return bool(re.search(r'#N\d+\.', text))

    def has_completion_indicators(self, text):
        return any(x in text for x in ['✅', '❌', 'remboursé', 'annulé'])

    def collect_inter_data(self, message):
        game_num = self.extract_game_number(message)
        if not game_num or game_num in self.collected_games: return
        
        cards = self.get_all_cards_in_first_group(message)
        if len(cards) < 2: return
        
        trigger = cards[0]
        # On collecte tout pour l'historique, mais on filtrera à l'analyse
        result_suit = None
        for card in cards:
            for suit, value in SUIT_TO_VALUE_MAP.items():
                if value in card:
                    result_suit = suit
                    break
            if result_suit: break
            
        if result_suit:
            self.inter_data.append({
                'game_number': game_num,
                'declencheur': trigger,
                'result_suit': result_suit,
                'timestamp': time.time()
            })
            self.collected_games.add(game_num)
            self._save_all_data()

    def analyze_and_set_smart_rules(self, chat_id=None, force_activate=False):
        self.inter_data = self._load_data('inter_data.json')
        groups = defaultdict(lambda: defaultdict(int))
        for entry in self.inter_data:
            trigger, result_val = entry['declencheur'], entry['result_suit']
            if not any(val in trigger for val in ['A', 'K', 'Q', 'J']):
                groups[result_val][trigger] += 1
        
        self.smart_rules = []
        for val in ["Q", "K", "J", "A"]:
            top = sorted(groups[val].items(), key=lambda x: x[1], reverse=True)[:5]
            for trigger, count in top:
                self.smart_rules.append({'trigger': trigger, 'predict': val, 'count': count})
        
        self.last_inter_update_time = time.time()
        self._save_all_data()

    def make_prediction(self, message, game_number):
        cards = self.get_first_two_cards_info(message)
        if not cards: return None
        trigger = cards[0]
        if any(val in trigger for val in ['A', 'K', 'Q', 'J']): return None
        
        if time.time() - self.last_prediction_time < self.prediction_cooldown: return None
        
        best_rule = next((r for r in self.smart_rules if r['trigger'] == trigger), None)
        if not best_rule: return None
        
        predicted = best_rule['predict']
        prediction = {
            'game_number': game_number, 'predicted_costume': predicted,
            'predicted_from_trigger': trigger, 'timestamp': time.time(),
            'status': 'pending', 'is_inter': True
        }
        self.predictions[game_number] = prediction
        self.last_prediction_time = time.time()
        self._save_all_data()
        return prediction

    def verify_prediction(self, text):
        game_num = self.extract_game_number(text)
        if not game_num: return None
        
        for pred_game in sorted(self.predictions.keys()):
            pred = self.predictions[pred_game]
            if pred.get('status') != 'pending': continue
            
            found, offset = False, 0
            for off in [0, 1, 2]:
                if game_num == pred_game + off:
                    cards = self.get_all_cards_in_first_group(text)
                    val_target = SUIT_TO_VALUE_MAP.get(pred['predicted_costume'], pred['predicted_costume'])
                    if any(val_target in c for c in cards):
                        pred['status'], pred['win_offset'], found, offset = 'won', off, True, off
                    elif off == 2:
                        pred['status'], found = 'lost', True
                    break
            
            if found:
                self._save_all_data()
                return {
                    'type': 'edit_message', 'game_number': pred_game,
                    'status': pred['status'], 'symbol': SYMBOL_MAP.get(offset if pred['status']=='won' else 'lost'),
                    'message_id': pred.get('message_id')
                }
        return None

    def get_inter_status(self):
        msg = f"🧠 **MODE INTER (Exclusif)**\n\nDonnées: {len(self.inter_data)} jeux\nRègles: {len(self.smart_rules)} (Top 5)\n\n"
        if self.smart_rules:
            by_suit = defaultdict(list)
            for r in self.smart_rules: by_suit[r['predict']].append(r)
            for s in ['Q', 'K', 'J', 'A']:
                if s in by_suit:
                    msg += f"**Cible {s}:**\n" + "\n".join([f"  • {r['trigger']} ({r['count']}x)" for r in by_suit[s]]) + "\n\n"
        else: msg += "⚠️ En attente de collecte...\n"
        return msg, {'inline_keyboard': [[{'text': '🔄 Analyser', 'callback_data': 'inter_apply'}]]}

    def check_and_send_reports(self):
        now = self.now()
        if now.hour not in [0, 6, 12, 18]: return
        key = f"{now.strftime('%Y-%m-%d')}_{now.hour}"
        if self.last_report_sent.get(key): return
        
        preds = [p for p in self.predictions.values() if p.get('status') in ['won', 'lost']]
        total = len(preds)
        wins = sum(1 for p in preds if p['status'] == 'won')
        rate = (wins/total*100) if total else 0
        
        msg = f"🎬 **BILAN**\n\nTotal: {total}\nSuccès: {wins} ({rate:.1f}%)\nIA: {self.get_inter_version()}"
        if self.telegram_message_sender and self.prediction_channel_id:
            self.telegram_message_sender(self.prediction_channel_id, msg)
            self.last_report_sent[key] = True
            self._save_all_data()

    def get_session_report_preview(self):
        preds = [p for p in self.predictions.values() if p.get('status') in ['won', 'lost']]
        total = len(preds)
        wins = sum(1 for p in preds if p['status'] == 'won')
        rate = (wins/total*100) if total else 0
        return f"📊 **STATS ACTUELLES**\n\nTotal: {total}\nSuccès: {wins} ({rate:.1f}%)\nMode: INTER Exclusif"

    def set_channel_id(self, channel_id, channel_type):
        if channel_type == 'source': self.target_channel_id = channel_id
        else: self.prediction_channel_id = channel_id
        self.config_data[f"{channel_type}_channel_id"] = channel_id
        self._save_data(self.config_data, 'channels_config.json')
        return True
