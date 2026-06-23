# Local Audiobook Dubbing Pipeline

Apple Silicon için yerel, uçtan uca YouTube sesli kitap çeviri ve dublaj
pipeline'ı. NLLB-200 ve XTTS-v2 PyTorch MPS üzerinde çalışır.
`faster-whisper`, CTranslate2'nin macOS MPS backend'i olmadığı için Apple
Silicon CPU üzerinde int8 çalışır. HTDemucs'taki bir Conv1d Metal'in mevcut
65.536 kanal sınırını aştığı için stem ayrımı güvenilir biçimde CPU'da yapılır.

## Kurulum

Gereksinimler:

- Apple Silicon Mac ve macOS
- Native arm64 CPython 3.11
- Homebrew FFmpeg
- Model indirmeleri için ilk çalıştırmada internet bağlantısı

```bash
brew install python@3.11 ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

XTTS-v2 ilk kullanımda Coqui Public Model License onayı isteyebilir. Lisansı
inceleyip arayüzden lisans durumunu seçmek zorunludur. Ticari olmayan CPML
seçimi gelir amaçlı YouTube veya uygulama yayını için `Yayına Hazır` onayı
vermez. Ticari yayın için Coqui ticari lisansı kullanıcı tarafından edinilmiş
ve arayüzde beyan edilmiş olmalıdır.

Kaynak içerik için de yayın hakkı beyanı zorunludur. Sistem telif/Content ID
atlatmak için tasarlanmamıştır; yalnızca kendi içeriğiniz, lisanslı/izinli
içerikler veya kamu malı eserler üretime alınır. Hak modu ve notu
`quality_report.json` içine yazılır; bu kayıt yayın öncesi denetim izi olarak
saklanır.

## Kullanım

```bash
python audiobook_pipeline.py \
  --url "https://www.youtube.com/watch?v=..." \
  --image "/absolute/path/cover.jpg" \
  --source_lang en \
  --target_langs tr es
```

Virgülle ayrılmış hedefler de kabul edilir:

```bash
python audiobook_pipeline.py \
  --url "https://www.youtube.com/watch?v=..." \
  --image "/absolute/path/cover.png" \
  --source_lang en \
  --target_langs tr,es
```

Çıktı:

```text
Book_Title_From_YouTube/
├── TR_dubbed.mp4
├── ES_dubbed.mp4
└── EN_original_bgm.wav
```

TTS doğal hızında tutulur. Bir cümle özgün zaman aralığından uzun sürerse
sonraki cümle ötelenir; ses hızlandırılmaz veya time-stretch uygulanmaz. BGM,
oluşan konuşmanın tam süresine kadar kesintisiz döngülenir.

## Profesyonel web arayüzü ve checkpoint sistemi

```bash
source .venv/bin/activate
uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Ardından `http://127.0.0.1:8000` adresini açın. Arayüz:

- YouTube URL, kaynak dil, hedef diller ve kapak görselini alır.
- Kaynak için `bana ait`, `lisanslı`, `yazılı izinli` veya `kamu malı` hak
  beyanı ister; güvenli hak beyanı yoksa üretim başlamaz.
- Kaynağı 10 dakikalık parçalara böler.
- Her stem, transkript, çeviri ve TTS segmentini diske atomik olarak kaydeder.
- Hata veya yeniden başlatma sonrasında yalnızca eksik parçadan devam eder.
- `Durdur`, çalışan worker ve Demucs alt sürecini anında dondurur.
- `Devam Et`, bellekteki aynı işlemi kaldığı noktadan sürdürür.
- `İptal`, çalışan süreç grubunu kapatır fakat tamamlanan checkpoint'leri
  korur; iş daha sonra yeniden devam ettirilebilir.
- Teknik günlük açıldığında otomatik yenilemelerde ve sayfa yeniden
  açıldığında açık kalır; yalnızca kullanıcı kapattığında kapanır.
- `Tamamen sil`, onaydan sonra iş kaydını, kaynak dosyaları, checkpoint'leri,
  günlükleri ve üretilmiş tüm çıktıları üretim kuyruğundan kalıcı olarak siler.
- Tamamlanmış veya durmuş projeye sonradan yeni hedef dil eklenebilir. Kaynak
  indirme, Demucs stemleri ve transkript yeniden hesaplanmaz; yalnızca eksik
  dilin çeviri, seslendirme, miks ve video dosyaları üretilir.
- Sıradan büyük harfli etiketler özel isim sayılmaz. Kısaltmalar, camelCase
  markalar, çok kelimeli isimler ve eser içinde tutarlı özel adlar korunur.
- Her dil için kayıpsız FLAC, YouTube'a uygun M4A ve MP4 üretir.
- Orijinal video modu seçilirse YouTube görüntüsü korunur, kaynak ses
  kaldırılır, dublajlı ses eklenir ve çeviri altyazısı üretilen dublajın
  gerçek zaman çizelgesine göre videoya yakılır.
- Çeviri kalite kapısı özel isim kaybı, iç koruma etiketi sızıntısı, bariz
  tekrar döngüsü ve çevrilmeden kalan Türkçe metadata etiketlerini yakalarsa
  TTS başlamadan işi durdurur.
- İlk kullanımda Whisper, NLLB ve XTTS model indirme/yükleme aşamalarını
  arayüzde açıkça gösterir.
- `quality_report.json`, doğal TTS, FLAC, YouTube M4A ve MP4 sürelerini ölçer;
  süre doğrulaması geçmeden iş tamamlandı sayılmaz.

Kurulum tamamlandıktan sonra Finder'dan `start_app.command` dosyasına çift
tıklayarak da arayüzü açabilirsiniz. Bu başlatıcı işlem boyunca Mac'in uykuya
geçmesini engeller.

24 saatlik bir işte Mac'in uykuya geçmesini önlemek için sunucuyu şöyle
başlatabilirsiniz:

```bash
caffeinate -dimsu uvicorn web_app:app --host 127.0.0.1 --port 8000
```

## Vercel landing deploy

Vercel deploy yalnızca satış/landing yüzeyi içindir. Yerel AI stüdyo; MPS,
Demucs, Whisper, TTS, FFmpeg ve uzun dosya işleme gerektirdiği için Mac
üzerinde veya kurumsal private worker olarak çalışır.

```bash
npx vercel
npx vercel --prod
```

Vercel CLI önceden kurulmuş ve login yapılmışsa aynı komutlar `npm run
deploy:preview` ve `npm run deploy:prod` olarak da çalışır.

Vercel route'ları:

- `/` ve `/landing`: yayıncı landing sayfası
- `/studio`: local/private stüdyo erişim açıklaması
- Local çalışmada `/studio`: gerçek üretim paneli

## Teknik doğruluk notu

Demucs çıktısı float32 WAV olarak tutulur ve ek kayıplı codec uygulanmaz.
Bununla birlikte kaynak ayırma, öğrenilmiş bir tahmin işlemidir; hiçbir stem
modeli matematiksel olarak kusursuz veya gerçek anlamda kayıpsız ayrım garantisi
veremez.
