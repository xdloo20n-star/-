"use strict";

const API = "/api/v1";
const POLL_INTERVAL = 1600;
const SELECTION_KEY = "sverka.comparison.v1";
const UNKNOWN = "не указано";
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const FILTER_NAMES = ["q", "metric_name", "has_issues", "document_id", "dataset_id", "task_id", "config_id"];
const ACTIVE_STATUSES = new Set(["queued", "processing"]);
const STATUS_LABELS = {
  queued: "В очереди", processing: "Извлекаем данные", completed: "Обработан", failed: "Ошибка извлечения",
};
const WORKSPACE = document.querySelector("#workspace");
const TRAY = document.querySelector("#compare-tray");
const NOTICE = document.querySelector("#notice");
const EXPORT_ALL = document.querySelector("#export-all");
const INTEGER_FORMAT = new Intl.NumberFormat("ru-RU");
const VALUE_FORMAT = new Intl.NumberFormat("ru-RU", { maximumSignificantDigits: 16 });
const DELTA_FORMAT = new Intl.NumberFormat("ru-RU", { maximumSignificantDigits: 12, signDisplay: "exceptZero" });
const DATE_FORMAT = new Intl.DateTimeFormat("ru-RU", { dateStyle: "medium", timeStyle: "short" });

let currentScreen;
let noticeTimer;
let uploadInProgress = false;
let selection = restoreSelection();

function element(tag, className, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of children) {
    if (child != null) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

function button(label, className, action) {
  const node = element("button", className, label);
  node.type = "button";
  node.addEventListener("click", action);
  return node;
}

function link(label, href, className) {
  const node = element("a", className, label);
  node.href = href;
  return node;
}

function route(name, entries = {}) {
  const query = new URLSearchParams(entries).toString();
  return `#${name}${query ? `?${query}` : ""}`;
}

function documentRoute(id, follow = false) {
  return route("document", follow ? { id, follow: "1" } : { id });
}

function resultRoute(id, from) {
  return route("result", from ? { id, from } : { id });
}

function resultsRoute(query = {}, page = 1) {
  const params = new URLSearchParams(query);
  params.delete("page");
  if (page > 1) params.set("page", String(page));
  return route("results", params);
}

function endpoint(resource, id) {
  return `/${resource}/${encodeURIComponent(id)}`;
}

function routeBack(params) {
  const target = params.get("from");
  return /^#(?:results|groups)(?:\?|$)/.test(target ?? "") ? target : "#results";
}

function withQuery(path, query) {
  const params = new URLSearchParams(query).toString();
  return params ? `${path}?${params}` : path;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(API + path, {
      ...options, headers: { Accept: "application/json", ...options.headers },
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Не удалось связаться с сервером. Проверьте соединение.");
  }
  const payload = await response.json().catch((error) => {
    if (error.name === "AbortError") throw error;
    return null;
  });
  if (!response.ok) throw new Error(payload?.error?.message || `Сервер вернул ошибку ${response.status}.`);
  if (!payload) throw new Error("Сервер вернул ответ без данных.");
  return payload;
}

function valueOrUnknown(value) {
  return value == null || value === "" ? UNKNOWN : value;
}

function formatUnit(unit) {
  return unit === "fraction" ? "доля" : unit;
}

function measured(measurement, formatter = VALUE_FORMAT) {
  return `${formatter.format(measurement.value)} ${formatUnit(measurement.unit)}`;
}

function rawValue(result) {
  if (result.raw_value == null) return UNKNOWN;
  const value = result.raw_value;
  if (!result.raw_unit || value.trimEnd().endsWith(result.raw_unit)) return value;
  return `${value}${result.raw_unit === "%" ? "" : " "}${result.raw_unit}`;
}

function plural(count, forms) {
  const lastTwo = count % 100;
  const last = count % 10;
  if (lastTwo >= 11 && lastTwo <= 14) return forms[2];
  if (last === 1) return forms[0];
  return last >= 2 && last <= 4 ? forms[1] : forms[2];
}

function totalLabel(count, forms) {
  return `${INTEGER_FORMAT.format(count)} ${plural(count, forms)}`;
}

function mockBadge(mode) {
  return mode === "mock" ? element("span", "badge badge-mock", "Синтетические данные") : null;
}

function statusBadge(status) {
  return element("span", `badge badge-status status-${status}`, element("span", "status-dot"), STATUS_LABELS[status]);
}

function pdfLink(fileUrl, label, page) {
  if (!fileUrl) return null;
  let url;
  try {
    url = new URL(fileUrl, window.location.origin);
  } catch {
    return null;
  }
  if (url.origin !== window.location.origin || !["http:", "https:"].includes(url.protocol)) return null;
  if (page != null) {
    if (!Number.isSafeInteger(page) || page < 1) return null;
    url.hash = `page=${page}`;
  }
  const anchor = link(label, url.href, "button button-secondary pdf-link");
  anchor.target = "_blank";
  anchor.rel = "noopener noreferrer";
  anchor.setAttribute("aria-label", `${label} в новой вкладке`);
  return anchor;
}

function sectionHeading(title, ...actions) {
  return element("div", "section-heading", element("h2", "", title), ...actions);
}

function screenHeading(title, back) {
  const heading = element("h1", "", title);
  heading.tabIndex = -1;
  const previous = back ? link(back.label, back.href, "back-link") : null;
  return element("div", "screen-heading", previous, heading);
}

function statePanel(title, message, action, kind = "") {
  const panel = element("div", `state${kind ? ` state-${kind}` : ""}`,
    element("p", "state-title", title), message ? element("p", "state-message", message) : null, action,
  );
  if (kind === "error") panel.setAttribute("role", "alert");
  if (kind === "loading") panel.setAttribute("role", "status");
  return panel;
}

function loadFailure(target, title, error, retry) {
  target.replaceChildren(statePanel(title, error.message, button("Повторить", "button button-secondary", retry), "error"));
}

function pageNumber(params) {
  const number = Number(params.get("page"));
  return Number.isSafeInteger(number) && number > 0 ? number : 1;
}

function pagination(data, makeRoute) {
  const last = Math.max(1, Math.ceil(data.total / data.page_size));
  if (last === 1 && data.page === 1) return null;
  const nav = element("nav", "pagination");
  nav.setAttribute("aria-label", "Страницы списка");
  const previous = data.page > 1 ? link("Назад", makeRoute(data.page - 1), "button button-secondary") : null;
  const next = data.page < last ? link("Далее", makeRoute(data.page + 1), "button button-secondary") : null;
  nav.append(element("span", "page-position", "Страница ", element("span", "mono", `${data.page} / ${last}`)),
    element("div", "page-actions", previous, next));
  return nav;
}

function notify(message, action) {
  clearTimeout(noticeTimer);
  NOTICE.replaceChildren(element("div", "notice-card", element("span", "", message), action,
    button("Закрыть", "notice-dismiss", () => NOTICE.replaceChildren()),
  ));
  if (!action) noticeTimer = setTimeout(() => {
    if (!NOTICE.contains(document.activeElement)) NOTICE.replaceChildren();
  }, 8000);
}

async function exportJSON(trigger, documentId) {
  trigger.disabled = true;
  trigger.setAttribute("aria-busy", "true");
  try {
    const payload = await request(withQuery("/export", documentId ? { document_id: documentId } : {}));
    const blob = new Blob([JSON.stringify(payload, null, 2), "\n"], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const download = link("", url);
    download.download = documentId ? `sverka-${documentId}.json` : "sverka.json";
    document.body.append(download);
    download.click();
    download.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    notify("Выгрузка JSON готова.");
  } catch (error) {
    notify(`Не удалось выгрузить JSON. ${error.message}`);
  } finally {
    trigger.disabled = false;
    trigger.removeAttribute("aria-busy");
  }
}

function documentExport(id) {
  return button("Выгрузить JSON", "button button-secondary", (event) => exportJSON(event.currentTarget, id));
}

function showDocuments(screen, params) {
  const page = pageNumber(params);
  const input = element("input", "visually-hidden");
  input.type = "file";
  input.accept = ".pdf,application/pdf";
  input.tabIndex = -1;
  input.setAttribute("aria-label", "Файл PDF");
  const choose = button("Выбрать PDF", "button button-primary", () => input.click());
  choose.disabled = uploadInProgress;
  const uploadStatus = element("div", "upload-status");
  uploadStatus.setAttribute("aria-live", "polite");
  const drop = element("section", "panel upload-panel",
    element("div", "document-symbol", element("span", "", "PDF")),
    element("div", "upload-copy", element("h2", "", "Загрузить статью"),
      element("p", "muted", "Выберите или перетащите сюда один файл PDF."), uploadStatus),
    choose, input,
  );
  drop.firstChild.setAttribute("aria-hidden", "true");

  async function upload(files) {
    if (!files.length || uploadInProgress) return;
    if (files.length > 1) {
      uploadStatus.replaceChildren(element("p", "text-error", "Загрузите один PDF за раз."));
      return;
    }
    const body = new FormData();
    body.append("file", files[0]);
    uploadInProgress = true;
    choose.disabled = true;
    uploadStatus.replaceChildren(element("p", "muted", "Загружаем ", element("bdi", "", files[0].name), "…"));
    try {
      const accepted = await request("/documents", { method: "POST", body });
      const target = documentRoute(accepted.document_id, true);
      if (!screen.signal.aborted) location.hash = target;
      else notify("PDF добавлен в очередь.", link("Открыть статус", target, "text-link"));
    } catch (error) {
      if (screen.signal.aborted) notify(`Не удалось загрузить PDF. ${error.message}`);
      else uploadStatus.replaceChildren(element("p", "text-error", error.message));
    } finally {
      uploadInProgress = false;
      const visibleChoose = WORKSPACE.querySelector(".upload-panel .button-primary");
      if (visibleChoose) visibleChoose.disabled = false;
    }
  }

  input.addEventListener("change", () => {
    upload(input.files);
    input.value = "";
  });
  drop.addEventListener("dragover", (event) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = uploadInProgress ? "none" : "copy";
    drop.classList.add("is-dragging");
  });
  drop.addEventListener("dragleave", (event) => {
    if (!drop.contains(event.relatedTarget)) drop.classList.remove("is-dragging");
  });
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("is-dragging");
    upload(event.dataTransfer.files);
  });

  const list = element("div", "panel list-panel");
  const count = element("span", "section-count");
  const refresh = button("Обновить", "button button-secondary button-small", load);
  WORKSPACE.append(screenHeading("Документы"), drop,
    element("section", "workspace-section", sectionHeading("Очередь документов", count, refresh), list));

  let loading = false;
  let generation = 0;
  let timer;
  screen.signal.addEventListener("abort", () => clearTimeout(timer), { once: true });

  async function load() {
    if (loading || screen.signal.aborted) return;
    loading = true;
    refresh.disabled = true;
    clearTimeout(timer);
    const token = ++generation;
    list.replaceChildren(statePanel("Загружаем документы…", null, null, "loading"));
    count.textContent = "";
    try {
      const data = await request(withQuery("/documents", { page }), { signal: screen.signal });
      if (screen.signal.aborted) return;
      count.textContent = totalLabel(data.total, ["документ", "документа", "документов"]);
      if (data.items.length === 0) {
        list.replaceChildren(data.total
          ? statePanel("На этой странице нет документов", null, link("К первой странице", "#documents", "button button-secondary"))
          : statePanel("Документов пока нет", "Загрузите статью, чтобы добавить её в очередь."));
        return;
      }
      const rows = element("ul", "document-list");
      const active = new Map();
      for (const doc of data.items) {
        const row = element("li", "document-row");
        renderDocumentRow(row, doc);
        rows.append(row);
        if (ACTIVE_STATUSES.has(doc.status)) active.set(doc.document_id, { row, status: doc.status });
      }
      list.replaceChildren(rows);
      const pages = pagination(data, (number) => route("documents", { page: number }));
      if (pages) list.append(pages);

      async function poll() {
        const updates = await Promise.allSettled([...active.keys()].map(async (id) => {
          const detail = await request(endpoint("documents", id), { signal: screen.signal });
          return detail.document;
        }));
        if (screen.signal.aborted || token !== generation) return;
        for (const update of updates) {
          if (update.status !== "fulfilled") continue;
          const doc = update.value;
          const item = active.get(doc.document_id);
          if (!item) continue;
          if (item.status !== doc.status) {
            renderDocumentRow(item.row, doc);
            item.status = doc.status;
          }
          if (!ACTIVE_STATUSES.has(doc.status)) active.delete(doc.document_id);
        }
        if (active.size) timer = setTimeout(poll, POLL_INTERVAL);
      }
      if (active.size) timer = setTimeout(poll, POLL_INTERVAL);
    } catch (error) {
      if (!screen.signal.aborted) loadFailure(list, "Не удалось загрузить документы", error, load);
    } finally {
      loading = false;
      refresh.disabled = false;
    }
  }

  load();
}

function renderDocumentRow(row, doc) {
  const previousFocus = row.contains(document.activeElement) ? document.activeElement : null;
  const filename = link(doc.original_filename, documentRoute(doc.document_id), "document-name");
  const date = new Date(doc.uploaded_at);
  const time = element("time", "mono", Number.isNaN(date.getTime()) ? UNKNOWN : DATE_FORMAT.format(date));
  time.dateTime = doc.uploaded_at;
  const action = doc.status === "completed"
    ? link("Результаты", resultsRoute({ document_id: doc.document_id }), "text-link document-action")
    : link("Открыть статус", documentRoute(doc.document_id), "text-link document-action");
  row.replaceChildren(element("div", "document-identity", filename,
    element("div", "document-meta", time, doc.corpus_id ? element("span", "mono", doc.corpus_id) : null),
    doc.failure ? element("p", "text-error document-error", doc.failure.message) : null),
  element("div", "document-status", statusBadge(doc.status), mockBadge(doc.extraction_mode)), action);
  if (previousFocus) (previousFocus.classList.contains("document-action") ? action : filename).focus();
}

function documentMessages(data) {
  const messages = [];
  const publication = data?.publication;
  const pages = data?.source_document?.unreadable_pages ?? [];
  if (publication?.in_scope === false) {
    messages.push(element("div", "callout callout-warning", element("strong", "", "Статья вне области Сверки"),
      element("p", "", valueOrUnknown(publication.exclusion_reason))));
  }
  if (pages.length) {
    messages.push(element("div", "callout callout-warning", element("strong", "", "Извлечение выполнено частично"),
      element("p", "", `${plural(pages.length, ["Не прочитана страница", "Не прочитаны страницы", "Не прочитаны страницы"])} `,
        element("span", "mono", pages.join(", ")), ".")));
  }
  return messages;
}

function showDocument(screen, params) {
  const id = params.get("id");
  const follow = params.get("follow") === "1";
  const body = element("section", "panel document-detail");
  const heading = screenHeading("Статус документа", { label: "Документы", href: "#documents" });
  WORKSPACE.append(heading, body);
  if (!id) {
    body.append(statePanel("Документ не указан", null, link("К документам", "#documents", "button button-secondary")));
    return;
  }

  let timer;
  let lastStatus;
  screen.signal.addEventListener("abort", () => clearTimeout(timer), { once: true });

  async function load(showLoading = false) {
    if (showLoading) body.replaceChildren(statePanel("Получаем статус…", null, null, "loading"));
    try {
      const { document: doc, data } = await request(endpoint("documents", id), { signal: screen.signal });
      if (screen.signal.aborted) return;
      if (follow && doc.status === "completed" && data?.publication?.in_scope === true) {
        location.replace(resultsRoute({ document_id: id }));
        return;
      }
      if (doc.status !== lastStatus || showLoading) {
        heading.querySelector("h1").textContent = doc.original_filename;
        document.title = `${doc.original_filename} · Сверка`;
        const actions = element("div", "action-group", pdfLink(doc.file_url, "Открыть PDF"));
        if (doc.status === "completed") actions.append(documentExport(id));
        if (doc.status === "completed" && data?.publication?.in_scope === true) {
          actions.prepend(link("К результатам", resultsRoute({ document_id: id }), "button button-primary"));
        }
        const details = element("div", "document-detail-content");
        details.setAttribute("aria-live", "polite");
        details.append(element("div", "badge-group", statusBadge(doc.status), mockBadge(doc.extraction_mode)),
          data?.publication?.title ? element("h2", "", data.publication.title) : "",
          ...documentMessages(data));
        if (ACTIVE_STATUSES.has(doc.status)) {
          const message = doc.status === "queued" ? "Документ ожидает обработки." : "Обработка PDF продолжается.";
          details.append(element("p", "muted", message, follow ? " Результаты откроются после обработки." : ""));
        }
        if (doc.status === "failed") details.append(statePanel("Не удалось обработать PDF",
          doc.failure?.message, link("Загрузить другой PDF", "#documents", "button button-secondary"), "error"));
        body.replaceChildren(details, actions);
        lastStatus = doc.status;
      }
      if (ACTIVE_STATUSES.has(doc.status)) timer = setTimeout(load, POLL_INTERVAL);
    } catch (error) {
      if (!screen.signal.aborted) loadFailure(body, "Не удалось получить статус", error, () => load(true));
    }
  }

  load(true);
}

function resultFilters(params) {
  const filters = new URLSearchParams();
  for (const name of FILTER_NAMES) {
    const value = params.get(name)?.trim();
    if (value) filters.set(name, value);
  }
  return filters;
}

function filterInput(name, label, filters, placeholder, type = "text") {
  const input = element("input", "");
  input.type = type;
  input.name = name;
  input.value = filters.get(name) ?? "";
  if (placeholder) input.placeholder = placeholder;
  if (name.endsWith("_id")) {
    input.pattern = "[A-Za-z0-9][A-Za-z0-9_\\-]*";
    input.spellcheck = false;
  }
  return element("label", "filter-field", element("span", "", label), input);
}

function renderFilters(filters, reload) {
  const issues = element("select", "");
  issues.name = "has_issues";
  for (const [value, title] of [["", "Все"], ["true", "Есть замечания"], ["false", "Без замечаний"]]) {
    issues.add(new Option(title, value));
  }
  issues.value = filters.get("has_issues") ?? "";

  const submit = element("button", "button button-primary", "Применить");
  submit.type = "submit";
  const reset = filters.size ? link("Сбросить", "#results", "text-link filter-reset") : null;
  const form = element("form", "panel filters",
    element("div", "filter-main", filterInput("q", "Поиск", filters, "Статья, датасет", "search"),
      filterInput("metric_name", "Метрика", filters, "Recall@5"),
      element("label", "filter-field", element("span", "", "Замечания"), issues), submit),
  );
  const extra = element("details", "filter-extra", element("summary", "", "Дополнительные фильтры"),
    element("div", "filter-grid", filterInput("document_id", "ID документа", filters),
      filterInput("dataset_id", "ID датасета", filters), filterInput("task_id", "ID задачи", filters),
      filterInput("config_id", "ID конфигурации", filters)),
  );
  extra.open = ["dataset_id", "task_id", "config_id"].some((name) => filters.has(name));
  form.append(element("div", "filter-footer", extra, reset));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const next = resultsRoute(resultFilters(new URLSearchParams(new FormData(form))));
    if (next === location.hash) reload();
    else location.hash = next;
  });
  return form;
}

function showResults(screen, params) {
  const filters = resultFilters(params);
  const page = pageNumber(params);
  const id = filters.get("document_id");
  const from = resultsRoute(filters, page);
  const scope = element("div", "document-scope");
  const count = element("span", "section-count");
  const list = element("div", "panel list-panel");
  const heading = screenHeading("Результаты");
  const section = element("section", "workspace-section", sectionHeading("Извлечённые результаты", count), list);
  WORKSPACE.append(heading);
  if (id) WORKSPACE.append(scope);
  WORKSPACE.append(renderFilters(filters, load), section);

  let loading = false;
  async function load() {
    if (loading || screen.signal.aborted) return;
    loading = true;
    const query = new URLSearchParams(filters);
    query.set("page", String(page));
    count.textContent = "";
    list.replaceChildren(statePanel("Загружаем результаты…", null, null, "loading"));
    try {
      const data = await request(withQuery("/results", query), { signal: screen.signal });
      if (screen.signal.aborted) return;
      count.textContent = totalLabel(data.total, ["результат", "результата", "результатов"]);
      if (data.items.length) {
        list.replaceChildren(resultTable(data.items, from));
      } else if (data.total > 0) {
        list.replaceChildren(statePanel("На этой странице нет результатов", null,
          link("К первой странице", resultsRoute(filters), "button button-secondary")));
      } else if ([...filters.keys()].some((key) => key !== "document_id")) {
        const reset = id ? resultsRoute({ document_id: id }) : "#results";
        list.replaceChildren(statePanel("По фильтрам ничего не найдено", "Измените запрос или сбросьте фильтры.",
          link("Сбросить фильтры", reset, "button button-secondary")));
      } else if (id) {
        list.replaceChildren(statePanel("У документа пока нет результатов", "Статус обработки доступен в карточке документа.",
          link("К документу", documentRoute(id), "button button-secondary")));
      } else {
        list.replaceChildren(statePanel("Результатов пока нет", "После обработки статьи здесь появятся извлечённые записи.",
          link("К документам", "#documents", "button button-secondary")));
      }
      const pages = pagination(data, (number) => resultsRoute(filters, number));
      if (pages && data.items.length) list.append(pages);
    } catch (error) {
      if (!screen.signal.aborted) loadFailure(list, "Не удалось загрузить результаты", error, load);
    } finally {
      loading = false;
    }
  }

  async function loadScope() {
    scope.replaceChildren(element("div", "scope-bar panel", element("span", "muted", "Загружаем документ…")));
    try {
      const { document: doc, data } = await request(endpoint("documents", id), { signal: screen.signal });
      if (screen.signal.aborted) return;
      scope.replaceChildren(element("div", "scope-bar panel",
        element("div", "scope-identity", element("span", "field-label", "Документ"),
          link(doc.original_filename, documentRoute(id), "document-name"),
          data?.publication?.title ? element("p", "muted", data.publication.title) : null),
        element("div", "action-group", statusBadge(doc.status), mockBadge(doc.extraction_mode), documentExport(id)),
      ), ...documentMessages(data));
      if (doc.failure) scope.append(element("div", "callout callout-error", doc.failure.message));
    } catch (error) {
      if (!screen.signal.aborted) scope.replaceChildren(element("div", "callout callout-error",
        element("p", "", "Не удалось загрузить карточку документа. ", error.message),
        button("Повторить", "button button-secondary button-small", loadScope)));
    }
  }

  load();
  if (id) loadScope();
}

function showGroups(screen) {
  const content = element("div", "groups-content");
  WORKSPACE.append(screenHeading("Сопоставимые группы"),
    element("p", "muted", "Здесь собраны результаты разных публикаций с совпавшими условиями оценки RAG."), content);
  async function load() {
    content.replaceChildren(statePanel("Подбираем группы…", null, null, "loading"));
    try {
      const { groups } = await request("/groups", { signal: screen.signal });
      if (screen.signal.aborted) return;
      if (!groups.length) {
        content.replaceChildren(statePanel("Пока нет подтверждённых межстатейных групп",
          "Для объединения должны совпасть задача, датасет и его версия, сплит, определение метрики, k, шкала, агрегация и протокол оценки.",
          link("К документам", "#documents", "button button-secondary")));
        return;
      }
      content.replaceChildren(...groups.map((group) => {
        const entries = group.results.map((item) => element("li", "group-result",
          link(item.publication_title || item.publication_id,
            resultRoute(item.result_id, "#groups"), "metric-link"),
          element("span", "muted", `${item.configuration_name || UNKNOWN} · ${item.dataset_name || UNKNOWN} · ${item.metric_name}`),
          element("span", "mono", item.normalized ? measured(item.normalized) : item.raw_value || UNKNOWN)));
        const members = element("ul", "group-results", ...entries);
        const first = group.results[0];
        const second = group.results[1];
        return element("section", "panel group-card",
          element("div", "section-heading", element("div", "section-title",
            element("h2", "", first.metric_name), element("span", "section-count", `${group.results.length} результатов`))),
          element("p", "muted", `${first.dataset_name || UNKNOWN} · ${group.group_id}`), members,
          second ? link("Сравнить две статьи", route("compare", {
            left: first.result_id, right: second.result_id, from: "#groups",
          }), "button button-secondary") : null);
      }));
    } catch (error) {
      if (!screen.signal.aborted) loadFailure(content, "Не удалось загрузить группы", error, load);
    }
  }
  load();
}

function extractionLabel(status) {
  return { absent: "В статье не указано", present_not_extractable: "Не удалось извлечь" }[status];
}

function rawCell(result) {
  const status = extractionLabel(result.extraction_status);
  return element("div", "value-stack", element("span", "mono raw-value", rawValue(result)),
    status ? element("span", "value-note", status) : null);
}

function issueCount(count) {
  const badge = element("span", `issue-count mono${count > 0 ? " has-issues" : ""}`, INTEGER_FORMAT.format(count));
  badge.setAttribute("aria-label", totalLabel(count, ["замечание", "замечания", "замечаний"]));
  return badge;
}

function tableCell(label, ...content) {
  const cell = element("td", "", ...content);
  cell.dataset.label = label;
  return cell;
}

function resultTable(items, from) {
  const labels = ["Выбор", "Метрика", "Конфигурация", "Датасет", "В статье", "Нормализовано", "Замечания"];
  const header = element("tr", "", ...labels.map((label) => {
    const cell = element("th", "", label);
    cell.scope = "col";
    return cell;
  }));
  const body = element("tbody", "");
  for (const row of items) {
    const result = row.result;
    body.append(element("tr", "",
      tableCell(labels[0], pickResult(row)),
      tableCell(labels[1], element("div", "metric-cell",
        link(result.metric_name, resultRoute(result.result_id, from), "metric-link"),
        result.metric_variant ? element("span", "value-note", result.metric_variant) : null,
        mockBadge(row.extraction_mode))),
      tableCell(labels[2], valueOrUnknown(row.configuration_name)),
      tableCell(labels[3], valueOrUnknown(row.dataset_name)),
      tableCell(labels[4], rawCell(result)),
      tableCell(labels[5], row.normalized ? element("span", "mono normalized-value", measured(row.normalized)) : element("span", "muted", UNKNOWN)),
      tableCell(labels[6], issueCount(row.issue_count)),
    ));
  }
  return element("table", "results-table", element("caption", "visually-hidden", "Таблица извлечённых результатов"),
    element("thead", "", header), body);
}

function restoreSelection() {
  try {
    const stored = JSON.parse(localStorage.getItem(SELECTION_KEY));
    if (!Array.isArray(stored)) return [];
    return stored.slice(0, 2).filter((item, index) => item && typeof item.id === "string"
      && ID_PATTERN.test(item.id) && typeof item.metric === "string"
      && typeof item.context === "string" && stored.findIndex((entry) => entry?.id === item.id) === index);
  } catch {
    return [];
  }
}

function updateSelection(next) {
  selection = next;
  try {
    localStorage.setItem(SELECTION_KEY, JSON.stringify(selection));
  } catch {
    // Comparison selection still works in the current page when storage is disabled.
  }
  for (const checkbox of WORKSPACE.querySelectorAll("[data-pick-id]")) {
    checkbox.checked = selection.some((entry) => entry.id === checkbox.dataset.pickId);
  }
  renderTray();
}

function pickResult(row, showLabel = false) {
  const entry = { id: row.result.result_id, metric: row.result.metric_name,
    context: [row.configuration_name, row.dataset_name].filter(Boolean).join(" / "), mode: row.extraction_mode };
  const checkbox = element("input", "pick-checkbox");
  checkbox.type = "checkbox";
  checkbox.dataset.pickId = entry.id;
  checkbox.checked = selection.some((item) => item.id === entry.id);
  checkbox.setAttribute("aria-label", `Выбрать для сравнения ${[entry.metric, entry.context, entry.id].filter(Boolean).join(", ")}`);
  checkbox.addEventListener("change", () => {
    if (!checkbox.checked) {
      updateSelection(selection.filter((item) => item.id !== entry.id));
    } else if (selection.length === 2) {
      checkbox.checked = false;
      notify("Уберите один из двух выбранных результатов, чтобы добавить другой.");
    } else {
      updateSelection([...selection, entry]);
    }
  });
  return showLabel ? element("label", "pick-label", checkbox, "Выбрать для сравнения") : checkbox;
}

function renderTray() {
  TRAY.hidden = !selection.length || currentScreen?.name === "compare";
  document.body.classList.toggle("has-tray", !TRAY.hidden);
  if (TRAY.hidden) {
    TRAY.replaceChildren();
    return;
  }
  const heading = element("div", "tray-title", element("span", "", "Сравнение"),
    element("span", "mono", `${selection.length} / 2`));
  const slots = element("div", "tray-slots");
  const from = currentScreen?.resultsBack ?? "#results";
  for (const [index, side] of ["A", "B"].entries()) {
    const entry = selection[index];
    const slot = element("div", "tray-slot", element("span", "side-key mono", side));
    if (entry) {
      const title = link(entry.context ? `${entry.metric} · ${entry.context}` : entry.metric, resultRoute(entry.id, from), "tray-result");
      title.title = title.textContent;
      const remove = button("×", "tray-remove", () => {
        updateSelection(selection.filter((item) => item.id !== entry.id));
        (TRAY.querySelector(".tray-remove") ?? WORKSPACE.querySelector("h1")).focus({ preventScroll: true });
      });
      remove.setAttribute("aria-label", `Убрать результат ${side} из сравнения`);
      slot.append(title, mockBadge(entry.mode), remove);
    } else {
      slot.append(element("span", "muted", "Выберите результат"));
    }
    slots.append(slot);
  }
  const compare = selection.length === 2
    ? link("Сравнить", route("compare", { left: selection[0].id, right: selection[1].id, from }), "button button-primary")
    : element("button", "button button-primary", "Сравнить");
  if (compare instanceof HTMLButtonElement) {
    compare.type = "button";
    compare.disabled = true;
  }
  TRAY.replaceChildren(element("div", "tray-inner", heading, slots, compare));
}

const CONFIG_FIELDS = {
  config_id: "ID конфигурации", rag_variant: "Вариант RAG", retrieval_method: "Метод поиска",
  retriever_model: "Модель поиска", embedding_model: "Модель эмбеддингов", embedding_version: "Версия эмбеддингов",
  chunking_method: "Разбиение текста", chunk_size: "Размер фрагмента", chunk_overlap: "Перекрытие фрагментов",
  chunk_unit: "Единица разбиения", index_type: "Тип индекса", vector_store: "Векторное хранилище", top_k: "Top K поиска",
  reranker_model: "Модель переранжирования", generator_model: "Модель генерации", generator_version: "Версия генератора",
  prompt_strategy: "Стратегия промпта", context_strategy: "Стратегия контекста", note: "Примечание",
};
const DATASET_FIELDS = {
  dataset_id: "ID датасета", dataset_name: "Название", dataset_version: "Версия датасета", language: "Язык",
  data_split: "Разбиение выборки", scientific_domain: "Предметная область", note: "Примечание",
};
const TASK_FIELDS = { task_id: "ID задачи", task_type: "Тип задачи", note: "Примечание" };
const EXPERIMENT_FIELDS = {
  experiment_id: "ID эксперимента", evaluation_protocol: "Протокол оценки", baseline_name: "Базовая линия",
  hardware: "Оборудование", latency_ms: "Задержка, мс", tokens_per_query: "Токенов на запрос",
  sample_size: "Размер выборки", note: "Примечание",
};
const RESULT_FIELDS = {
  metric_name: "Метрика", metric_variant: "Вариант метрики", raw_value: "Запись в статье", raw_unit: "Единица в статье",
  reported_value: "Значение", reported_scale: "Исходная шкала", aggregation: "Агрегация", metric_k: "Параметр K",
  metric_direction: "Направление метрики", sample_size: "Размер выборки", note: "Примечание",
};
const EVIDENCE_FIELDS = {
  result: RESULT_FIELDS, configuration: CONFIG_FIELDS, dataset: DATASET_FIELDS, task: TASK_FIELDS, experiment: EXPERIMENT_FIELDS,
  publication: { title: "Название статьи", year: "Год публикации", authors: "Авторы", doi: "DOI" },
};
const ENTITY_LABELS = {
  result: "Результат", configuration: "Конфигурация", dataset: "Датасет", task: "Задача",
  experiment: "Эксперимент", publication: "Публикация",
};
const VERIFICATION = {
  text_match: "Текст совпал", ocr_match: "Текст совпал после OCR",
  locator_only: "Требует сверки", unverified: "Требует сверки",
};

function fieldList(entries, className = "field-list") {
  return element("dl", className, ...entries.map(([label, value]) =>
    element("div", "field-pair", element("dt", "", label),
      element("dd", typeof value === "number" ? "mono" : "", valueOrUnknown(value)))));
}

function entityDetails(title, entity, labels) {
  const details = element("details", "entity-details", element("summary", "", title));
  const fields = Object.entries(labels).filter(([key]) => entity?.[key] != null);
  details.append(fields.length ? fieldList(fields.map(([key, label]) => [label, entity[key]]))
    : element("p", "muted", UNKNOWN));
  return details;
}

function sourceCard(evidence, context, index) {
  const mock = context.mode === "mock";
  const target = context.allowPageLinks ? pdfLink(context.fileUrl, "Открыть страницу PDF", evidence.page) : null;
  const verification = mock ? mockBadge("mock") : element("span",
    `badge ${["text_match", "ocr_match"].includes(evidence.verification) ? "badge-success" : "badge-warning"}`,
    VERIFICATION[evidence.verification] ?? "Требует сверки");
  const entity = ENTITY_LABELS[evidence.entity_type] ?? evidence.entity_type;
  const field = Object.hasOwn(EVIDENCE_FIELDS[evidence.entity_type] ?? {}, evidence.field)
    ? EVIDENCE_FIELDS[evidence.entity_type][evidence.field] : evidence.field;
  const coordinates = [["Таблица", evidence.table], ["Строка", evidence.table_row], ["Столбец", evidence.table_column],
    ["Рисунок", evidence.figure], ["Раздел", evidence.section]].filter(([, value]) => value != null);
  const card = element("article", "source-card",
    element("div", "source-head",
      element("div", "source-title", element("h3", "", `Источник ${index + 1}`),
        element("span", "muted", field ? `${entity} · ${field}` : entity)), verification),
    element("div", "source-body",
      element("div", "source-page", element("span", "", mock ? "Стр. в примере" : "Страница"),
        element("strong", "mono", valueOrUnknown(evidence.page))),
      element("div", "source-contents",
        evidence.fragment ? element("blockquote", "source-excerpt", evidence.fragment)
          : element("p", "source-excerpt muted", "Цитата недоступна."),
        coordinates.length ? fieldList(coordinates, "source-coordinates") : null)),
    target ? element("div", "source-footer", target) : null);
  card.tabIndex = -1;
  return card;
}

function renderIssues(issues, sourceNodes, from) {
  const section = element("section", "workspace-section review-checks", sectionHeading("Проверки", element("span", "section-count mono", issues.length)));
  const panel = element("div", "panel");
  section.append(panel);
  if (!issues.length) {
    panel.append(element("p", "checks-clear", element("span", "check-mark", "✓"), "Замечаний нет"));
    return section;
  }
  const list = element("ul", "issue-list");
  for (const issue of issues) {
    const sources = issue.evidence_ids.map((id) => {
      const source = sourceNodes.get(id);
      if (!source) return element("span", "muted mono", `Недоступен: ${id}`);
      return button(`К источнику ${source.number}`, "text-link source-jump", () => {
        source.node.focus({ preventScroll: true });
        source.node.scrollIntoView({ block: "nearest" });
      });
    });
    const related = issue.result_ids.map((id) => link(id, resultRoute(id, from), "text-link mono"));
    list.append(element("li", "issue-item",
      element("div", "issue-title", element("code", "mono", issue.check_name),
        element("span", `badge ${issue.severity === "warning" ? "badge-warning" : "badge-neutral"}`,
          issue.severity === "warning" ? "Предупреждение" : "Сведения")),
      element("p", "", issue.description),
      sources.length ? element("div", "issue-links", ...sources) : null,
      related.length ? element("div", "issue-links", element("span", "field-label", "Связанные результаты"), ...related) : null));
  }
  panel.append(list);
  return section;
}

function resultOverview(detail) {
  const result = detail.result;
  const origin = result.result_origin === "cited" ? "Цитируемый результат" : "Результат этой статьи";
  return element("section", "panel result-overview",
    element("div", "overview-identity",
      element("div", "badge-group", element("span", "badge badge-neutral", origin), mockBadge(detail.extraction_mode)),
      element("h2", "", valueOrUnknown(detail.publication.title)),
      fieldList([["Конфигурация", detail.configuration_name], ["Датасет", detail.dataset_name], ["Задача", detail.task_type],
        ...(result.result_origin === "cited" && result.cited_publication_id ? [["ID цитируемой публикации", result.cited_publication_id]] : [])],
      "overview-fields")),
    element("div", "overview-value",
      element("span", "field-label", "Запись в статье"), rawCell(result),
      detail.normalized ? element("div", "normalized-detail", "Нормализовано ", element("span", "mono", measured(detail.normalized))) : null,
      detail.baseline ? element("div", "normalized-detail", "Базовый результат ", element("span", "mono", measured(detail.baseline))) : null),
  );
}

function renderResultDetail(detail, docDetail, docError, from, reload) {
  const doc = docDetail?.document;
  const mode = detail.extraction_mode === "mock" || doc?.extraction_mode === "mock" ? "mock" : detail.extraction_mode;
  const context = { mode, fileUrl: doc?.file_url, allowPageLinks: mode === "python" && doc?.extraction_mode === "python" };
  const related = new Set(detail.issues.flatMap((issue) => issue.evidence_ids));
  const evidence = new Map(detail.evidence.map((source) => [source.evidence_id, source]));
  for (const source of docDetail?.data?.evidence ?? []) {
    if (related.has(source.evidence_id) && !evidence.has(source.evidence_id)) evidence.set(source.evidence_id, source);
  }
  const sources = element("section", "workspace-section source-column", sectionHeading("Источники", element("span", "section-count mono", evidence.size)));
  if (docError) {
    sources.append(element("div", "callout callout-error",
      element("p", "", "Карточка документа недоступна. ", docError.message),
      button("Повторить", "button button-secondary button-small", reload)));
  }
  const sourceNodes = new Map();
  const sourceRank = (source) => {
    if (source.entity_type !== "result" || source.entity_id !== detail.result.result_id) return 2;
    return source.field === "raw_value" || source.field == null ? 0 : 1;
  };
  const orderedSources = [...evidence].sort(([, left], [, right]) => sourceRank(left) - sourceRank(right));
  for (const [id, source] of orderedSources) {
    const card = sourceCard(source, context, sourceNodes.size);
    sourceNodes.set(id, { node: card, number: sourceNodes.size + 1 });
    sources.append(card);
  }
  if (!evidence.size) sources.append(element("div", "panel", statePanel("Источник не сохранён", "Для этого результата нет фрагментов или локаторов.")));
  const conditions = element("section", "workspace-section", sectionHeading("Условия эксперимента"),
    element("div", "panel entity-list", entityDetails("Конфигурация", detail.configuration, CONFIG_FIELDS),
      entityDetails("Датасет", detail.dataset, DATASET_FIELDS), entityDetails("Задача", detail.task, TASK_FIELDS),
      entityDetails("Эксперимент", detail.experiment, EXPERIMENT_FIELDS)));
  const actions = element("div", "review-actions", link("К документу", documentRoute(detail.document_id), "text-link"),
    documentExport(detail.document_id), pickResult(detail, true));
  return element("div", "review", resultOverview(detail), actions,
    element("div", "review-body", sources, element("div", "context-column", renderIssues(detail.issues, sourceNodes, from), conditions)));
}

function showResult(screen, params) {
  const id = params.get("id");
  const from = routeBack(params);
  const heading = screenHeading("Результат", { label: "Результаты", href: from });
  const content = element("div", "result-detail");
  WORKSPACE.append(heading, content);
  if (!id) {
    content.append(element("div", "panel", statePanel("Результат не указан", null,
      link("К результатам", from, "button button-secondary"))));
    return;
  }

  async function load() {
    content.replaceChildren(element("div", "panel", statePanel("Загружаем результат…", null, null, "loading")));
    try {
      const detail = await request(endpoint("results", id), { signal: screen.signal });
      if (screen.signal.aborted) return;
      heading.querySelector("h1").textContent = detail.result.metric_name;
      document.title = `${detail.result.metric_name} · Сверка`;
      const included = new Set(detail.evidence.map((item) => item.evidence_id));
      const needsDocument = detail.extraction_mode === "python"
        || detail.issues.some((issue) => issue.evidence_ids.some((evidenceId) => !included.has(evidenceId)));

      let doc;
      let docError;
      if (needsDocument) {
        try {
          doc = await request(endpoint("documents", detail.document_id), { signal: screen.signal });
        } catch (error) {
          docError = error;
        }
      }
      if (!screen.signal.aborted) content.replaceChildren(renderResultDetail(detail, doc, docError, from, load));
    } catch (error) {
      if (!screen.signal.aborted) loadFailure(content, "Не удалось загрузить результат", error, load);
    }
  }

  load();
}

const COMPARISON_LABELS = {
  comparable: "Результаты сопоставимы",
  not_comparable: "Условия различаются",
  insufficient_information: "Недостаточно сведений для сравнения",
};
const CRITERION_LABELS = {
  task: "Задача", dataset: "Датасет", dataset_version: "Версия датасета", data_split: "Разбиение выборки",
  metric_definition: "Определение метрики", metric_k: "Параметр K", scale: "Шкала", aggregation: "Агрегация",
  evaluation_protocol: "Протокол оценки",
};
const CRITERION_STYLES = {
  match: ["success", "Совпадает"], mismatch: ["error", "Различается"], missing: ["warning", "Нет сведений"],
  not_applicable: ["neutral", "Не применяется"],
};

function comparisonSide(side, id, mode, detail, value, preferred, from) {
  const saved = selection.find((entry) => entry.id === id);
  const metric = detail?.result.metric_name ?? saved?.metric ?? id;
  return element("section", `panel comparison-side${preferred ? " preferred-side" : ""}`,
    element("div", "comparison-side-header", element("span", "side-key mono", side),
      element("span", "field-label", `Результат ${side}`), mockBadge(mode),
      preferred ? element("span", "badge badge-success", "Предпочтительный") : null),
    element("h2", "", link(metric, resultRoute(id, from), "metric-link")),
    detail ? fieldList([["Конфигурация", detail.configuration_name], ["Датасет", detail.dataset_name]])
      : saved?.context ? element("p", "muted", saved.context) : null,
    detail ? element("div", "comparison-value", value != null ? "В статье " : "", rawCell(detail.result)) : null,
    value != null ? element("p", "comparison-number mono", value) : null);
}

function criterionList(criteria) {
  const list = element("ul", "criteria-list");
  for (const criterion of criteria) {
    const [style, label] = CRITERION_STYLES[criterion.status];
    list.append(element("li", "criterion",
      element("div", "criterion-title", element("strong", "", CRITERION_LABELS[criterion.criterion] ?? criterion.criterion),
        element("span", `badge badge-${style}`, label)), element("p", "", criterion.message)));
  }
  return list;
}

function renderComparison(comparison, left, right, from) {
  const numeric = comparison.numeric;
  const preferredLabels = { left: "Предпочтительнее A", right: "Предпочтительнее B", tie: "Значения равны" };
  const decisionStyle = { comparable: "success", not_comparable: "error", insufficient_information: "warning" }[comparison.status];
  const decision = element("section", `panel comparison-decision decision-${decisionStyle}`,
    element("div", "decision-text", element("span", "field-label", "Итог сравнения"),
      element("h2", "", COMPARISON_LABELS[comparison.status]),
      numeric && preferredLabels[numeric.preferred] ? element("p", "", preferredLabels[numeric.preferred]) : null),
    numeric ? element("div", "delta", element("span", "field-label", "Разница B − A"),
      element("strong", "mono", measured({ value: numeric.delta, unit: numeric.unit }, DELTA_FORMAT))) : null);
  const criteria = comparison.criteria;
  const applicable = ["mismatch", "missing", "match"].flatMap((status) => criteria.filter((item) => item.status === status));
  const notApplicable = criteria.filter((item) => item.status === "not_applicable");
  return element("div", "comparison",
    decision,
    element("div", "comparison-pair",
      comparisonSide("A", comparison.left_result_id, comparison.left_extraction_mode, left,
        numeric ? measured({ value: numeric.left_value, unit: numeric.unit }) : null, numeric?.preferred === "left", from),
      comparisonSide("B", comparison.right_result_id, comparison.right_extraction_mode, right,
        numeric ? measured({ value: numeric.right_value, unit: numeric.unit }) : null, numeric?.preferred === "right", from)),
    element("section", "workspace-section", sectionHeading("Критерии сравнения"),
      element("div", "panel", applicable.length ? criterionList(applicable) : null,
        notApplicable.length ? element("details", "extra-criteria", element("summary", "", "Неприменимые критерии ",
          element("span", "mono muted", notApplicable.length)), criterionList(notApplicable)) : null,
        !criteria.length ? statePanel("Критерии не указаны") : null)));
}

function showComparison(screen, params) {
  const left = params.get("left");
  const right = params.get("right");
  const from = routeBack(params);
  const content = element("div", "comparison-content");
  WORKSPACE.append(screenHeading("Сравнение результатов", { label: "Изменить выбор", href: from }), content);
  if (!left || !right || left === right) {
    content.append(element("div", "panel", statePanel("Нужны два разных результата", "Отметьте две записи в списке.",
      link("К результатам", from, "button button-secondary"))));
    return;
  }

  async function load() {
    content.replaceChildren(element("div", "panel", statePanel("Сравниваем результаты…", null, null, "loading")));
    try {
      const options = { signal: screen.signal };
      const [comparison, leftDetail, rightDetail] = await Promise.all([
        request("/comparisons", { ...options, method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ left_result_id: left, right_result_id: right }) }),
        request(endpoint("results", left), options).catch(() => null),
        request(endpoint("results", right), options).catch(() => null),
      ]);
      if (!screen.signal.aborted) content.replaceChildren(renderComparison(comparison, leftDetail, rightDetail, from));
    } catch (error) {
      if (!screen.signal.aborted) loadFailure(content, "Не удалось сравнить результаты", error, load);
    }
  }

  load();
}

const SCREENS = {
  documents: { render: showDocuments, title: "Документы" }, document: { render: showDocument, title: "Статус документа" },
  results: { render: showResults, title: "Результаты" }, groups: { render: showGroups, title: "Сопоставимые группы" },
  result: { render: showResult, title: "Результат" },
  compare: { render: showComparison, title: "Сравнение результатов" },
};

function renderScreen(focus) {
  const fragment = location.hash.slice(1);
  const separator = fragment.indexOf("?");
  const requested = separator === -1 ? fragment : fragment.slice(0, separator);
  const query = separator === -1 ? "" : fragment.slice(separator + 1);
  const name = Object.hasOwn(SCREENS, requested) ? requested : "documents";
  const params = new URLSearchParams(query);
  if (currentScreen) currentScreen.controller.abort();
  const controller = new AbortController();
  currentScreen = { name, controller, signal: controller.signal };
  if (name === "results") currentScreen.resultsBack = resultsRoute(resultFilters(params), pageNumber(params));
  if (name === "result") currentScreen.resultsBack = routeBack(params);
  document.title = `${SCREENS[name].title} · Сверка`;
  const documents = name === "documents" || name === "document";
  const groups = name === "groups";
  const results = !documents && !groups;
  document.querySelector("#nav-documents").toggleAttribute("data-current", documents);
  document.querySelector("#nav-results").toggleAttribute("data-current", results);
  document.querySelector("#nav-groups").toggleAttribute("data-current", groups);
  document.querySelector("#nav-documents").setAttribute("aria-current", documents ? "location" : "false");
  document.querySelector("#nav-results").setAttribute("aria-current", results ? "location" : "false");
  document.querySelector("#nav-groups").setAttribute("aria-current", groups ? "location" : "false");
  WORKSPACE.replaceChildren();
  SCREENS[name].render(currentScreen, params);
  WORKSPACE.removeAttribute("aria-busy");
  renderTray();
  if (focus) {
    window.scrollTo(0, 0);
    WORKSPACE.querySelector("h1")?.focus({ preventScroll: true });
  }
}

document.querySelector(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  WORKSPACE.focus();
});
EXPORT_ALL.addEventListener("click", () => exportJSON(EXPORT_ALL));
window.addEventListener("hashchange", () => renderScreen(true));
renderScreen(false);
