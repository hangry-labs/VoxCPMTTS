# Speech Quality Details

Run `20261007-184048` across 24 common languages.
Text is normalized with Unicode NFKC, case folding, and removal of non-letter/non-number characters.
Similarity uses a normalized sequence ratio; exact means the normalized transcript equals the prompt.
Generation is stochastic; repeated samples characterize the distribution rather than paired identical outputs.

## native

| Mode | Judged | Exact | Mean similarity | Minimum | Errors |
|---|---:|---:|---:|---:|---:|
| design | 240 | 57.92% | 96.47% | 37.66% | 0 |
| clone | 72 | 6.94% | 93.98% | 42.52% | 0 |

### Languages

| Language | Judged | Exact | Mean similarity | Minimum | Errors |
|---|---:|---:|---:|---:|---:|
| Arabic | 13 | 38.46% | 93.60% | 69.23% | 0 |
| Chinese | 13 | 69.23% | 98.99% | 90.91% | 0 |
| Danish | 13 | 0.00% | 88.90% | 77.23% | 0 |
| Dutch | 13 | 38.46% | 97.00% | 84.03% | 0 |
| English | 13 | 84.62% | 99.90% | 99.05% | 0 |
| Finnish | 13 | 15.38% | 91.45% | 37.66% | 0 |
| French | 13 | 61.54% | 98.07% | 80.00% | 0 |
| German | 13 | 92.31% | 99.95% | 99.33% | 0 |
| Greek | 13 | 0.00% | 85.83% | 74.29% | 0 |
| Hindi | 13 | 61.54% | 94.50% | 77.02% | 0 |
| Indonesian | 13 | 53.85% | 97.23% | 86.87% | 0 |
| Italian | 13 | 69.23% | 99.69% | 97.61% | 0 |
| Japanese | 13 | 61.54% | 97.97% | 91.43% | 0 |
| Korean | 13 | 76.92% | 97.09% | 87.01% | 0 |
| Malay | 13 | 53.85% | 96.59% | 80.43% | 0 |
| Polish | 13 | 30.77% | 95.83% | 89.36% | 0 |
| Portuguese | 13 | 53.85% | 98.63% | 90.53% | 0 |
| Russian | 13 | 53.85% | 97.00% | 86.69% | 0 |
| Spanish | 13 | 38.46% | 98.33% | 90.91% | 0 |
| Swedish | 13 | 0.00% | 87.69% | 42.52% | 0 |
| Tagalog | 13 | 0.00% | 93.51% | 86.32% | 0 |
| Thai | 13 | 69.23% | 97.51% | 87.91% | 0 |
| Turkish | 13 | 30.77% | 97.64% | 94.12% | 0 |
| Vietnamese | 13 | 53.85% | 98.70% | 94.21% | 0 |

### Hardest Calls

| Call | Expected | Transcript | Similarity |
|---|---|---|---:|
| finnish_design_02_r2 | Laitoin avaimet turvalliseen paikkaan, liian turvalliseen paikkaan. | Jenä, laitit toimimaan, mutta kun koforesaanik, hen orkaan sin keo, em, ei, se, the chase, tai, tai, lamparessu broadside, ei | 37.66% |
| swedish_clone_01_r2 | Hej, det här är VoxCPMTTS från Hangry Labs. Vi bygger lokala, lättkörda röstverktyg så att människor kan skapa tal privat, offline och på sitt eget språk. | Hej, här är Vaxis PMTTS från Hangry Labs. De kallar Cole "Lecturer" i Strasbourg taget, så en intern studiekompetens behövde beda offline under den sista tråden. | 42.52% |
| arabic_design_01_r5 | القهوة هذا الصباح تفاوضت معي قبل أن تمنحني النشاط. | قل كواحدة تصبحت فوطة معي قبل أن تمنحين النشاط | 69.23% |
| swedish_design_02_r3 | Jag lade nycklarna på ett säkert ställe, lite för säkert. | Jag glädd nötklänaren på att säkert ställa lite för säkert. | 73.68% |
| greek_design_02_r5 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Αυτός είναι ο αβαλακλιδιάς ασφαλες μέρος, τόσο ασφαλες μουχάτικαν. | 74.29% |
| greek_design_01_r2 | Ο πρωινός καφές διαπραγματεύτηκε σκληρά, αλλά τελικά βοήθησε. | ποιοτική ο προηνός καφέ διαπραματευτικής κληρά αλλα τελικά βοηθήσει | 75.68% |
| greek_design_02_r3 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | αμα χαριέμαι πάλι τα κλινιά σας φαλες μέρος τόσα φαλες που χάθηκαν | 76.92% |
| hindi_clone_01_r2 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते यह हैंगरी लैब्स का वॉक सी पी एम टी टी एस है हम स्थानीय और आसानी से चलने वाले वॉइस टूल बनाते हैं ताकि लोग निजी रूप से ऑफलाइन और अपनी भाषा में भाषण बना सकें. | 77.02% |
| danish_design_02_r4 | Jeg lagde nøglerne et sikkert sted, og nu er stedet alt for sikkert. | et er en nøgleret sikker sted og nu er sted alt forsigtigt | 77.23% |
| greek_design_02_r1 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Εμβιρείταν θα κλειντιάσει ασφαλες μέρος, τόσα ασφαλες που χάθηκαν. | 78.10% |
| finnish_design_02_r5 | Laitoin avaimet turvalliseen paikkaan, liian turvalliseen paikkaan. | jopaust me tieden asuut laitoi avaimet turvallisen paikkaa liian turvallisen paikkaa heti | 78.83% |
| hindi_clone_01_r1 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह hungrylabs का walk CPM TTS है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें. | 79.53% |
| hindi_clone_01_r3 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह hungry labs का walk cp mtts है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें. | 79.53% |
| swedish_design_02_r1 | Jag lade nycklarna på ett säkert ställe, lite för säkert. | Jag glädjat nylagen på ett säkert ställe lite för säkert. | 79.57% |
| french_design_01_r5 | Le café du matin a négocié ferme, puis il a accepté de m'aider. | Le café du matin a négocié fermes puis a essayé de le mener. | 80.00% |
| malay_design_02_r1 | Saya letak kunci di tempat selamat, mungkin terlalu selamat. | Ya, datang kunci di tepat selamat kita lalu selamat. | 80.43% |
| danish_design_02_r2 | Jeg lagde nøglerne et sikkert sted, og nu er stedet alt for sikkert. | alleg det nøglerne er et sikret sted og nu er stedet alt forsigtigt | 80.73% |
| greek_design_01_r5 | Ο πρωινός καφές διαπραγματεύτηκε σκληρά, αλλά τελικά βοήθησε. | Ο προηγούμενος καθέςια πραγματευτικής κληριά αλλά τελικά βοήθησε. | 80.73% |
| arabic_clone_01_r1 | مرحبا، هذا VoxCPMTTS من Hangry Labs. نبني أدوات صوت محلية وسهلة التشغيل حتى يتمكن الناس من إنشاء الكلام بخصوصية، دون اتصال، وبلغتهم. | مرحبا، هذا ووكسي بمتياس من هانجري لابز. نبني أدوات صوت محلي وسهلة التشغيل حتى يتمكن الناس من إنشاء الكلام بخصوصية دون اتصال. وبلغتهم. | 80.75% |
| greek_design_02_r4 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Ε, βαλα τα κλειδία σε ασφαλές μέρος, τόσο ασφαλές βγουα θα κάνα. | 80.81% |

## nano

| Mode | Judged | Exact | Mean similarity | Minimum | Errors |
|---|---:|---:|---:|---:|---:|
| design | 240 | 57.92% | 96.80% | 38.00% | 0 |
| clone | 72 | 9.72% | 94.91% | 74.70% | 0 |

### Languages

| Language | Judged | Exact | Mean similarity | Minimum | Errors |
|---|---:|---:|---:|---:|---:|
| Arabic | 13 | 30.77% | 95.86% | 71.58% | 0 |
| Chinese | 13 | 69.23% | 98.56% | 90.91% | 0 |
| Danish | 13 | 0.00% | 89.68% | 84.96% | 0 |
| Dutch | 13 | 46.15% | 98.15% | 90.91% | 0 |
| English | 13 | 100.00% | 100.00% | 100.00% | 0 |
| Finnish | 13 | 15.38% | 93.24% | 76.92% | 0 |
| French | 13 | 53.85% | 98.97% | 91.67% | 0 |
| German | 13 | 92.31% | 99.95% | 99.33% | 0 |
| Greek | 13 | 0.00% | 88.61% | 76.60% | 0 |
| Hindi | 13 | 76.92% | 95.52% | 74.70% | 0 |
| Indonesian | 13 | 53.85% | 99.24% | 96.70% | 0 |
| Italian | 13 | 53.85% | 98.95% | 92.47% | 0 |
| Japanese | 13 | 61.54% | 97.12% | 87.72% | 0 |
| Korean | 13 | 76.92% | 96.93% | 86.45% | 0 |
| Malay | 13 | 38.46% | 95.45% | 76.80% | 0 |
| Polish | 13 | 30.77% | 95.74% | 87.13% | 0 |
| Portuguese | 13 | 53.85% | 98.74% | 94.74% | 0 |
| Russian | 13 | 69.23% | 98.18% | 89.27% | 0 |
| Spanish | 13 | 38.46% | 99.05% | 98.00% | 0 |
| Swedish | 13 | 15.38% | 88.30% | 38.00% | 0 |
| Tagalog | 13 | 7.69% | 94.82% | 90.72% | 0 |
| Thai | 13 | 53.85% | 96.96% | 89.89% | 0 |
| Turkish | 13 | 30.77% | 96.99% | 89.11% | 0 |
| Vietnamese | 13 | 53.85% | 97.71% | 93.33% | 0 |

### Hardest Calls

| Call | Expected | Transcript | Similarity |
|---|---|---|---:|
| swedish_design_02_r3 | Jag lade nycklarna på ett säkert ställe, lite för säkert. | jag inte jag inte jag gör det skönt att jag tycker en spaceföretag | 38.00% |
| arabic_design_01_r4 | القهوة هذا الصباح تفاوضت معي قبل أن تمنحني النشاط. | كره هذا الصباح تفضلت معي قبل أن تمنحني النشاط ها ها ها أستئ إلى أستئ | 71.58% |
| hindi_clone_01_r1 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह हैंग्री लैब्स का Vox CPMTTS है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें. | 74.70% |
| greek_design_02_r2 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Εμάνα τα κλίδια σας φαλες μέρος, τους φαλες που χάθηκαν. | 76.60% |
| malay_design_02_r1 | Saya letak kunci di tempat selamat, mungkin terlalu selamat. | saya tak kunci di tempat selamat mungkin terlalu selamat ambitami anda seakredit mampat | 76.80% |
| finnish_design_02_r1 | Laitoin avaimet turvalliseen paikkaan, liian turvalliseen paikkaan. | Ollaan oikein avaimet turvallisempiäkkä liian turvallisempiäkkä. | 76.92% |
| greek_design_02_r4 | Έβαλα τα κλειδιά σε ασφαλές μέρος, τόσο ασφαλές που χάθηκαν. | Εβαλα τα κλινιά σε ασφάλεις μέρος, τόσο ασφάλεις ποχάτκα. | 79.17% |
| swedish_design_02_r2 | Jag lade nycklarna på ett säkert ställe, lite för säkert. | jag kladdi nylagen på ett säkert ställe lite för säkert | 82.61% |
| hindi_clone_01_r2 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह hungry labs का Vox CPM TTS है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें. | 83.53% |
| hindi_clone_01_r3 | नमस्ते, यह Hangry Labs का VoxCPMTTS है। हम स्थानीय और आसानी से चलने वाले वॉयस टूल बनाते हैं, ताकि लोग निजी रूप से, ऑफलाइन, और अपनी भाषा में भाषण बना सकें। | नमस्ते, यह hungrylabs का Vox CPM TTS है। हम स्थानीय और आसानी से चलने वाले voice tool बनाते हैं, ताकि लोग निजी रूप से offline और अपनी भाषा में भाषण बना सकें. | 83.53% |
| greek_design_01_r2 | Ο πρωινός καφές διαπραγματεύτηκε σκληρά, αλλά τελικά βοήθησε. | Ο προηνός καφές διαπραματεύτηκε σκληρά, αλλά τελικά βοήθησε. Α, η σεντούνθηκε. | 84.48% |
| danish_design_02_r5 | Jeg lagde nøglerne et sikkert sted, og nu er stedet alt for sikkert. | Jeg har lagt den nøglerne i et sikret sted, og nu er stedet alt forsigtigt. | 84.96% |
| swedish_design_02_r5 | Jag lade nycklarna på ett säkert ställe, lite för säkert. | Jag gläder nycelarna på det säget ställe lite för säkt | 85.71% |
| danish_clone_01_r1 | Hej, dette er VoxCPMTTS fra Hangry Labs. Vi bygger lokale, nemme stemmeværktøjer, så folk kan skabe tale privat, offline og på deres eget sprog. | Hag, det er det af Vox CMPMTTS for Hengry Labs. Vi bygger lokale nemme stemmevaktører, så folk kan skabe tale privat, offline og på deres arbejdsbog. | 86.32% |
| swedish_clone_01_r3 | Hej, det här är VoxCPMTTS från Hangry Labs. Vi bygger lokala, lättkörda röstverktyg så att människor kan skapa tal privat, offline och på sitt eget språk. | Hej, detta är Vox i PMTTS från Hengry Labs. Vi bygger lokala, lättcydda röstverktjugsatmanager kan skapa tal privat offline och på sit egetsbråk. | 86.42% |
| korean_clone_01_r1 | 안녕하세요, Hangry Labs의 VoxCPMTTS입니다. 우리는 사람들이 자신의 언어로, 개인적으로, 오프라인에서 음성을 만들 수 있도록 로컬에서 쉽게 실행되는 음성 도구를 만듭니다. | 안녕하세요 핸드글래프스의 먹스 CPMTTS입니다. 우리는 사람들이 자신의 언어로 개인적으로 오프라인에서 음성을 만들 수 있도록 로컬에서 쉽게 실행되는 음성 도구를 만듭니다. | 86.45% |
| danish_clone_01_r2 | Hej, dette er VoxCPMTTS fra Hangry Labs. Vi bygger lokale, nemme stemmeværktøjer, så folk kan skabe tale privat, offline og på deres eget sprog. | Hej, det er det er Vox PMTTS fra hungry Labs. Vi bygger lokale nemmes demvegttoyer, så folk kan skabe tale privat, offline og på deres adesbror. | 86.46% |
| greek_clone_01_r1 | Γεια σας, αυτό είναι το VoxCPMTTS από τα Hangry Labs. Δημιουργούμε τοπικά και εύκολα εργαλεία φωνής, ώστε οι άνθρωποι να παράγουν ομιλία ιδιωτικά, εκτός σύνδεσης και στη γλώσσα τους. | Γιασας, αυτό είναι το Vox C, την TTS από τα Henry Labs. Τη μιοργούμε τοπικά και εύκολη εργαλεία φωνής, ώστε οι άνθρωποι να παράγουν ομηλιά ιδιοτικά εκτός σύνταξης και στιγμιότυπους. | 86.49% |
| danish_design_02_r1 | Jeg lagde nøglerne et sikkert sted, og nu er stedet alt for sikkert. | Jeg lagt nøglerne et sikret sted og nu er stedet alt forsigtigt. | 86.79% |
| korean_clone_01_r2 | 안녕하세요, Hangry Labs의 VoxCPMTTS입니다. 우리는 사람들이 자신의 언어로, 개인적으로, 오프라인에서 음성을 만들 수 있도록 로컬에서 쉽게 실행되는 음성 도구를 만듭니다. | 안녕하세요, 행글랍스의 박시 PMTTS입니다. 우리는 사람들이 자신의 언어로 개인적으로 오프라인에서 음성을 만들 수 있도록 로컬에서 쉽게 실행되는 음성 도구를 만듭니다. | 86.84% |
