document.querySelectorAll('.symptom-range').forEach(input => {
  const output = input.parentElement.querySelector('output');
  input.addEventListener('input', () => output.value = input.value);
});
const result = document.querySelector('#result');
if (result) result.scrollIntoView({behavior: 'smooth', block: 'center'});
