# Speech Quality Details

Run `20261029-070100` across 24 common languages.
Text is normalized with Unicode NFKC, case folding, and removal of non-letter/non-number characters.
Similarity uses a normalized sequence ratio; exact means the normalized transcript equals the prompt.
Each call uses a stable seed derived from its call ID; repeated cases still measure within-runtime consistency.

## nano

| Mode | Judged | Exact | Mean similarity | Minimum | Errors |
|---|---:|---:|---:|---:|---:|
| design | 240 | 53.33% | 96.18% | 23.66% | 0 |
| clone | 72 | 6.94% | 95.05% | 79.53% | 0 |

### Languages

| Language | Judged | Exact | Mean similarity | Minimum | Errors |
|---|---:|---:|---:|---:|---:|
| Arabic | 13 | 15.38% | 95.60% | 80.75% | 0 |
| Chinese | 13 | 76.92% | 99.56% | 97.56% | 0 |
| Danish | 13 | 0.00% | 88.54% | 81.08% | 0 |
| Dutch | 13 | 46.15% | 92.79% | 23.66% | 0 |
| English | 13 | 92.31% | 99.94% | 99.20% | 0 |
| Finnish | 13 | 0.00% | 93.92% | 86.73% | 0 |
| French | 13 | 38.46% | 98.13% | 93.88% | 0 |
| German | 13 | 84.62% | 99.84% | 98.92% | 0 |
| Greek | 13 | 0.00% | 88.35% | 83.17% | 0 |
| Hindi | 13 | 69.23% | 90.84% | 34.33% | 0 |
| Indonesian | 13 | 53.85% | 95.94% | 71.23% | 0 |
| Italian | 13 | 53.85% | 98.92% | 94.62% | 0 |
| Japanese | 13 | 53.85% | 97.22% | 85.38% | 0 |
| Korean | 13 | 76.92% | 97.05% | 87.01% | 0 |
| Malay | 13 | 38.46% | 98.02% | 91.84% | 0 |
| Polish | 13 | 23.08% | 97.25% | 88.24% | 0 |
| Portuguese | 13 | 53.85% | 95.92% | 71.43% | 0 |
| Russian | 13 | 76.92% | 98.19% | 88.97% | 0 |
| Spanish | 13 | 38.46% | 98.37% | 93.07% | 0 |
| Swedish | 13 | 7.69% | 94.98% | 84.21% | 0 |
| Tagalog | 13 | 0.00% | 95.04% | 90.38% | 0 |
| Thai | 13 | 30.77% | 91.37% | 51.39% | 0 |
| Turkish | 13 | 30.77% | 97.44% | 91.43% | 0 |
| Vietnamese | 13 | 61.54% | 98.80% | 94.21% | 0 |

### Hardest Calls

| Call | Expected | Transcript | Similarity |
|---|---|---|---:|
| dutch_design_02_r3 | Ik legde mijn sleutels veilig weg, zo veilig dat niemand ze meer kent. | Ooit lees naar kados, een klees voor een slees. | 23.66% |
| hindi_design_02_r1 | मैंने चाबियाँ सुरक्षित जगह रखीं, शायद बहुत ही सुरक्षित। | एफ एफ एफ एफ ले या सेवन को रखना तो कभी कभी करना ठीक है हर चाबियां सुरक्षित एक जगह रखें एक बहुत ही सुरक्षित एलिजांबिका शहीद देश का दान है एक एटम हेपोवन एवं डैमेंट कनेक्ट है एक एटम टेकन डैमेंट डिंटी वेंट डिंटी डिं | 34.33% |
| thai_design_02_r1 | ฉันวางกุญแจไว้ในที่ปลอดภัย ปลอดภัยเกินไปจนหาไม่เจอ | แต่วันนั้นเอาตั้งค่านั้นสั่งใครน้ำปังทีนั้นสั่งเข็นกุ้งแดกง่ายเกิดประวัติในจากชั้นความนิ่งใส่ชั้นวางคุณแจ็วไว้ในที่ปลอดภัยปลอดภัยเกินไปจนหาไม่เจอ | 51.39% |
| indonesian_design_01_r3 | Kopi pagi berunding cukup keras, tetapi akhirnya mau membantu. | Kopi pagi berunding cukup keras, tetapi akhirnya mau membantu. Untarik nyima tiki raya di jempak makan di sana nih. | 71.23% |
| portuguese_design_02_r1 | Guardei as chaves em um lugar seguro, seguro demais. | Porque deixa a vizinha lugar seguro seguro demais | 71.43% |
| hindi_clone_01_r2 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह hungry labs का walk CPM TTS है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें। | 79.53% |
| arabic_clone_01_r3 | مرحبا، هذا VoxCPMTTS من Hangry Labs. نبني أدوات صوت محلية وسهلة التشغيل حتى يتمكن الناس من إنشاء الكلام بخصوصية، دون اتصال، وبلغتهم. | مرحبا هذا بوكسي ب.م.ت.ت.س من هانجري لابز. نبني أدوات صوت محليه وسهلة التشغيل حتى يتمكن الناس من إنشاء الكلام بخصوصية دون اتصال وبلغتهم. | 80.75% |
| thai_clone_01_r1 | สวัสดี นี่คือ VoxCPMTTS จาก Hangry Labs เราสร้างเครื่องมือเสียงที่รันในเครื่องและใช้งานง่าย เพื่อให้ผู้คนสร้างเสียงพูดได้อย่างเป็นส่วนตัว ออฟไลน์ และในภาษาของตนเอง | ตัวเด็ดสำหรับอัพเป็นไอพีดีมทททสจาก Hangry Labs เราสร้างเครื่องมือเสียงที่รันในเครื่องและใช้งานง่ายเพื่อให้ผู้คนสร้างเสียงพูดได้อย่างเป็นส่วนตัว offline และในภาษาของตนเอง | 80.78% |
| danish_design_02_r4 | Jeg lagde nøglerne et sikkert sted, og nu er stedet alt for sikkert. | Jeg lader nogle lærere i et sikret sted og nu er jeg sted alt forsikret. | 81.08% |
| portuguese_design_02_r4 | Guardei as chaves em um lugar seguro, seguro demais. | eu estava em um lugar seguro seguro demais | 83.12% |
| greek_design_02_r3 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Εβάλατα κληριδιάσας φαλλησμέρος, τόσος φαλλησις που χάθηκαν. | 83.17% |
| hindi_clone_01_r1 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह hungrylabs का Vox CPM TTS है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें. | 83.53% |
| hindi_clone_01_r3 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह hungry labs का Vox CPM TTS है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें. | 83.53% |
| danish_clone_01_r2 | Hej, dette er VoxCPMTTS fra Hangry Labs. Vi bygger lokale, nemme stemmeværktøjer, så folk kan skabe tale privat, offline og på deres eget sprog. | Kjære, det er vokse PMTTS fra Henry Labs. Vi bygger lokale nemme stemmevektorer, så folk kan skabe tale privat, offline og på deres adskab. | 84.07% |
| greek_design_02_r5 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Εύβαλα τα κλινιάσα σφαλες μέρος τωσ σφαλες που χάθηκαν. | 84.21% |
| swedish_design_02_r3 | Jag lade nycklarna på ett säkert ställe, lite för säkert. | Ja, lada nycklarna på ett särskilt ställe, lite för särskilt. | 84.21% |
| greek_design_01_r2 | Ο πρωινός καφές διαπραγματεύτηκε σκληρά, αλλά τελικά βοήθησε. | Ο προηνός καφές διαπραγματευτικής κληριά, αλλά τελικά βοηθήσει. | 84.91% |
| danish_clone_01_r1 | Hej, dette er VoxCPMTTS fra Hangry Labs. Vi bygger lokale, nemme stemmeværktøjer, så folk kan skabe tale privat, offline og på deres eget sprog. | Hej, det der er vokse CPM TDS for Hungary Labs. Vi bygger lokale nemme stemmevektorer, så folk kan skabe tale privat offline og på deres adskab. | 85.22% |
| japanese_clone_01_r3 | こんにちは、Hangry Labs の VoxCPMTTS です。私たちはローカルで簡単に動かせる音声ツールを作り、誰もが自分の言語で、プライベートに、オフラインで音声を生成できるようにします。 | こんにちは、ヘングリラブスのボックス C P M T T S です。私たちはローカルで簡単に動かせる音声ツールを作り、誰もが自分の言語でプライベート、オフラインで音声を生成できるようにします。 | 85.38% |
| greek_design_02_r2 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Είμαι λατακλιδιάς ασφαλες μέρος τόσα ασφαλες που χάθηκαν | 85.71% |
