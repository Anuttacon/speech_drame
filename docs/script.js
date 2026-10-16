const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
function activateTab(tab) {
  for (const candidate of tabs) {
    const selected = candidate === tab;
    candidate.setAttribute('aria-selected', String(selected));
    candidate.tabIndex = selected ? 0 : -1;
    document.getElementById(candidate.getAttribute('aria-controls')).hidden = !selected;
  }
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', (event) => {
    let destination;
    if (event.key === 'ArrowRight') destination = tabs[(index + 1) % tabs.length];
    if (event.key === 'ArrowLeft') destination = tabs[(index - 1 + tabs.length) % tabs.length];
    if (event.key === 'Home') destination = tabs[0];
    if (event.key === 'End') destination = tabs[tabs.length - 1];
    if (destination) {
      event.preventDefault();
      activateTab(destination);
      destination.focus();
    }
  });
});
const copyButton = document.getElementById('copy-citation');
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'BibTeX copied.';
    copyButton.textContent = 'Copied';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Use your browser’s Copy command.';
  }
});
