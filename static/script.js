document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('mri-file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const fileLabel = document.querySelector('.file-label');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');

    const originalImageElem = document.getElementById('original-image');
    const segmentedImageElem = document.getElementById('segmented-image');
    const maskOverlayElem = document.getElementById('mask-overlay');
    const predictionTextElem = document.getElementById('prediction-text').querySelector('span');
    const confidenceTextElem = document.getElementById('confidence-text').querySelector('span');
    
    let fileHandle = null;

    fileInput.addEventListener('change', (e) => {
        fileHandle = e.target.files[0];
        if (fileHandle) {
            fileLabel.textContent = fileHandle.name;
            analyzeBtn.disabled = false;
        } else {
            fileLabel.textContent = 'Choose an Image';
            analyzeBtn.disabled = true;
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!fileHandle) {
            alert('Please choose an image file first.');
            return;
        }

        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        loader.classList.remove('hidden');

        const reader = new FileReader();
        reader.readAsDataURL(fileHandle);
        reader.onload = async () => {
            const base64StringWithData = reader.result;
            const base64String = base64StringWithData.split(',')[1];
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: base64String }),
                });

                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                
                const data = await response.json();
                displayResults(data, base64StringWithData);

            } catch (error) {
                console.error('Error:', error);
                showError();
            } finally {
                loader.classList.add('hidden');
            }
        };
        reader.onerror = (error) => {
            console.error('File reading error:', error);
            showError();
            loader.classList.add('hidden');
        };
    });

    function displayResults(data, originalImageBase64) {
        originalImageElem.src = originalImageBase64;
        segmentedImageElem.src = originalImageBase64;
        maskOverlayElem.src = `data:image/png;base64,${data.mask}`;
        predictionTextElem.textContent = data.classification;
        confidenceTextElem.textContent = data.confidence;
        resultsSection.classList.remove('hidden');
    }

    function showError() {
        errorSection.classList.remove('hidden');
    }
});
