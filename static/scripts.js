let selectedSymptoms = [];

        function filterSymptoms() {
            let input = document.getElementById("symptomInput").value;
            if (input.length === 0) {
                document.getElementById("dropdownContent").style.display = "none";
                return;
            }

            fetch('/suggest_symptoms', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ partial_symptom: input })
            })
            .then(response => response.json())
            .then(data => {
                let dropdownContent = document.getElementById("dropdownContent");
                dropdownContent.innerHTML = '';
                data.forEach(symptom => {
                    let symptomDiv = document.createElement("div");
                    symptomDiv.innerText = symptom;
                    symptomDiv.onclick = () => addSymptom(symptom);
                    dropdownContent.appendChild(symptomDiv);
                });
                dropdownContent.style.display = 'block';
            });
        }

        function addSymptom(symptom) {
            if (!selectedSymptoms.includes(symptom) && selectedSymptoms.length < 4) {
                selectedSymptoms.push(symptom);
                document.getElementById("symptomInput").value = '';
                document.getElementById("dropdownContent").style.display = "none";
                updateSelectedSymptoms();
            } else if (selectedSymptoms.length >= 4) {
                alert('You can select up to 4 symptoms only.');
            }
        }

        function updateSelectedSymptoms() {
            let selectedSymptomsDiv = document.getElementById("selectedSymptoms");
            selectedSymptomsDiv.innerHTML = '<h3>Selected Symptoms:</h3>';
            selectedSymptoms.forEach(symptom => {
                let symptomDiv = document.createElement("div");
                symptomDiv.innerText = symptom;
                selectedSymptomsDiv.appendChild(symptomDiv);
            });
        }

        function getPrediction() {
            if (selectedSymptoms.length !== 4) {
                alert('Please select exactly 4 symptoms.');
                return;
            }

            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symptoms: selectedSymptoms })
            })
            .then(response => response.json())
            .then(data => {
                let resultDiv = document.getElementById("result");
                resultDiv.innerHTML = `<h3>Predicted Disease: ${data.disease}</h3>`;
                resultDiv.innerHTML += `<h3>Recommended Medication: ${data.medication}</h3>`;
            });
        }