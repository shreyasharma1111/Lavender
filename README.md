# Lavender: Women's Wellness Companion
Lavender is a holistic, AI-powered platform designed to support women's health and wellness with intelligent tools, emotional support, multilingual access, and community engagement. Below are the key features of the application:


🔐 User Authentication


* Secure signup/login system with hashed passwords (Bcrypt).

* JWT-based session management with HTTP-only secure cookies.



🏠 Personalized Dashboard

* Displays logged-in user’s email.

* Central hub to access all wellness tools.



🧠 AI Symptom Checker

* Supports 10+ disease prediction models (PCOS, diabetes, thyroid, various cancers, etc.).

* Uses trained .pkl models to calculate disease likelihood.

* Suggests natural remedies, diet, and exercise routines tailored to predictions.



📓 Journal

* Private journaling space for emotional or medical reflections.

* Timestamped entries stored securely and displayed chronologically.



🗓️ Period Tracker

* Log period start date and cycle length.

* Automatically calculates:

* Ovulation date

* Next expected period

* All data stored for tracking and analysis.




🎤 Voice Note Transcription (Whisper AI)

* Record and upload voice notes.

* Uses OpenAI's Whisper model to transcribe audio into text.

* Saves transcript and original audio filename for future reference.




🧘 AI-Based Phase Tips

* Detects current menstrual phase based on logs:

* Menstrual, Follicular, Ovulation, or Luteal.

* Provides phase-specific diet, rest, and fitness suggestions.




💬 Emotional Coach

* Delivers a new daily affirmation to boost emotional health.

* Randomly chosen and saved so all users see the same quote daily.




🧪 Weekly Health Challenge (Gamified)

* Rotating quiz-style health trivia.

* Users earn points and unlock:

          🥉 Bronze Achiever (50 pts)
          
          🥈 Silver Star (100 pts)
          
          🥇 Gold Champion (200 pts)




🧑‍⚕️ Doctor & Volunteer Dashboards


* Doctors:

    View non-doctor users.
    
    Submit medical advice stored in DB.



* Volunteers:

    Submit health resources (videos, links, descriptions).
    
    View existing submissions.




📚 Rural Education + Voice Submissions

* Users can upload community materials (with transcription).

* Useful for low-literacy areas, enabling content in voice + text formats.




🎨 Drawing Canvas

* Express creatively via an in-browser drawing tool.

* Saves Base64 image data with timestamps.




🎮 AI Game Quiz

* Simple multiple-choice health quiz.

* Encourages health literacy via points and feedback.




📤 Export to PDF

* One-click export of:

* Journal entries

* Period logs

* Neatly formatted into a downloadable PDF.




🌐 Multilingual Support

* Interface available in:

* 🇬🇧 English, 🇮🇳 Hindi, 🇫🇷 French, 🇪🇸 Spanish, 🇵🇰 Urdu, 🇩🇪 German

* Dynamically change via /set_language/<lang>.

* Powered by Flask-Babel with .po/.mo files in translations/.




🔔 Background Health Notifications


Runs as background daemon threads:

    💧 Hourly water intake reminders (8 AM – 10 PM)
    
    🌸 Period cycle prediction alerts
    
    🌞 Daily affirmation messages




📁 Clean Folder Structure

* app/ for main app logic

* templates/ for HTML Jinja2 views

* static/ for CSS, JS, images

* ai_models/ for all ML models

* translations/ for multilingual support






Note- some model training (.pkl) files and datasets (.csv) could not be uploaded due to GitHub's file size limitations. You can generate those specific model training files by training your own model by using the provided code, and download the datasets from Kaggle.
