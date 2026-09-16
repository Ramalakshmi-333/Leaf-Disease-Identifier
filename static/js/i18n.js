window.I18n = (() => {
  const storageKey = "agrpro.language";
  let current = "en";

  function getLanguage() {
    return current;
  }

  function setLanguage(language) {
    if (!translations[language]) return false;
    current = language;
    localStorage.setItem(storageKey, language);
    document.documentElement.lang = language;
    document.documentElement.dir = "ltr";
    return true;
  }

  function savedLanguage() {
    const saved = localStorage.getItem(storageKey);
    return translations[saved] ? saved : null;
  }

  function text(key, vars = {}) {
    const parts = key.split(".");
    let value = translations[current];
    for (const part of parts) value = value?.[part];
    if (typeof value !== "string") return key;
    return value.replace(/\{(\w+)\}/g, (_, name) => vars[name] ?? `{${name}}`);
  }

  function disease(label) {
    return translations[current].diseaseNames[label] || label;
  }

  function value(group, value) {
    return translations[current][`${group}Values`]?.[value] || value || "--";
  }

  function apply(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = text(element.dataset.i18n);
    });
    root.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
      element.placeholder = text(element.dataset.i18nPlaceholder);
    });
    root.querySelectorAll("[data-i18n-title]").forEach((element) => {
      element.title = text(element.dataset.i18nTitle);
    });
  }

  function init(language) {
    current = translations[language] ? language : "en";
    document.documentElement.lang = current;
    document.documentElement.dir = "ltr";
    apply();
  }

  return { getLanguage, setLanguage, savedLanguage, text, disease, value, apply, init };
})();