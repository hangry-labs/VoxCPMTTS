window.VOICE_EXAMPLES = [
  {
    "name": "Auto Multilingual TTS",
    "mode": "Auto",
    "language": "30 languages",
    "text": "VoxCPM2 can synthesize multilingual speech directly from text without a fixed speaker inventory.",
    "api": "{\"text\":\"Hello from VoxCPMTTS.\",\"language\":\"English\",\"output_format\":\"mp3\"}"
  },
  {
    "name": "Voice Design",
    "mode": "Design",
    "language": "Prompt controlled",
    "text": "Describe a new voice with natural language attributes such as age, tone, emotion, pace, accent, or dialect.",
    "api": "{\"text\":\"A designed VoxCPM2 sample.\",\"control\":\"young female, warm and gentle, slightly smiling\",\"output_format\":\"mp3\"}"
  },
  {
    "name": "Reference Clone",
    "mode": "Clone",
    "language": "Any supported language",
    "text": "Provide a short reference audio clip to preserve timbre, or add its transcript for transcript-guided cloning.",
    "api": "{\"text\":\"A cloned VoxCPM2 sample.\",\"ref_audio\":\"/data/ref.wav\",\"ref_text\":\"Transcript of the reference audio.\",\"output_format\":\"mp3\"}"
  }
];
