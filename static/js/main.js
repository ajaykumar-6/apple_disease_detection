$(document).ready(function () {
    let currentLang = 'en';
    let langData = {};

    // 1. Load UI translations
    $.getJSON('/static/languages.json', function(data) {
        langData = data;
        updateUI('en');
    });

    function updateUI(lang) {
        currentLang = lang;
        $('[data-key]').each(function() {
            let key = $(this).data('key');
            if (langData[lang] && langData[lang][key]) {
                $(this).text(langData[lang][key]);
            }
        });
    }

    // 2. Language Selector - Handles switching even after results are shown
    $('#language-selector').change(function() {
        const selectedLang = $(this).val();
        updateUI(selectedLang);

        // If a result is already visible, re-run prediction to translate the result card
        if ($('#result').is(':visible') && $('#imageUpload').val() !== "") {
            $('#btn-predict').click();
        }
    });

    // 3. Image Preview & Reset Button Text for ALL Languages
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('.image-section').show(); 
                $('#result').hide(); // Hide old results
                
                // RESET BUTTON TO "PREDICT" (In whichever language is currently selected)
                $('#btn-predict').prop('disabled', false);
                if(langData[currentLang]) {
                    // This pulls "Predict Disease", "बीमारी की जांच करें", or "వ్యాధిని గుర్తించండి"
                    $('#btn-predict').text(langData[currentLang]['predict_btn']);
                }
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        readURL(this);
    });

    // 4. Prediction Logic
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        form_data.append('lang', currentLang);

        $(this).prop('disabled', true);
        if(langData[currentLang]) {
            $(this).text(langData[currentLang]['predicting']);
        }
        
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn().html(data);
                
                // CHANGE BUTTON TO "PREDICT AGAIN" (In current language)
                if(langData[currentLang]) {
                    $('#btn-predict').prop('disabled', false).text(langData[currentLang]['predict_again']);
                }
            },
            error: function () {
                $('.loader').hide();
                $('#btn-predict').prop('disabled', false).text("Error");
            }
        });
    });
});
// --- DARK MODE LOGIC ---
// --- THEME LOGIC (DEFAULT: LIGHT) ---
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');

// 1. Force 'light' as default if nothing is saved in localStorage
let currentTheme = localStorage.getItem('theme');

if (!currentTheme) {
    currentTheme = 'light';
    localStorage.setItem('theme', 'light');
}

// 2. Apply the theme on page load
function applyTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        if (themeIcon) themeIcon.textContent = '☀️';
    } else {
        document.documentElement.setAttribute('data-theme', 'light');
        document.documentElement.removeAttribute('data-theme'); // Ensure clean light mode
        if (themeIcon) themeIcon.textContent = '🌙';
    }
}

applyTheme(currentTheme);

// 3. Handle Toggle Click
themeToggle.addEventListener('click', () => {
    let theme = document.documentElement.getAttribute('data-theme');
    
    if (theme === 'dark') {
        applyTheme('light');
        localStorage.setItem('theme', 'light');
    } else {
        applyTheme('dark');
        localStorage.setItem('theme', 'dark');
    }
});