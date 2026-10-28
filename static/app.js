
const $ = (selector) => document.querySelector(selector);
const api = "/api";

let tickets = [];
let fields = [];
let activeTicket = null;


// ========================================
// API REQUESTS
// ========================================

async function request(path, options = {}) {
    const response = await fetch(api + path, {
        headers: {
            "Content-Type": "application/json"
        },
        ...options
    });

    const data = await response.json();

    if (!response.ok) {
        throw Error(data.error || "Request failed");
    }

    return data;
}


// ========================================
// HELPER FUNCTIONS
// ========================================

function esc(value = "") {
    return String(value).replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}

function slug(value) {
    return value.toLowerCase().replaceAll(" ", "-");
}

function notify(message) {
    const toast = $("#toast");

    toast.textContent = message;
    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 2800);
}

function stamp(dateString) {
    return new Date(dateString).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric"
    });
}


// ========================================
// RENDER TICKET TABLE
// ========================================

function render() {
    const status = $("#statusFilter").value;

    const shownTickets = tickets.filter((ticket) => {
        return status === "all" || ticket.status === status;
    });

    $("#ticketRows").innerHTML = shownTickets.length
        ? shownTickets.map((ticket) => `
            <tr data-id="${ticket.id}">
                <td class="ticket-id">
                    #T-${String(ticket.id).padStart(4, "0")}
                </td>

                <td>${esc(ticket.customer_name)}</td>

                <td class="subject-cell">
                    ${esc(ticket.subject)}
                </td>

                <td>
                    <span class="priority-${slug(ticket.priority)}">
                        ${esc(ticket.priority)}
                    </span>
                </td>

                <td>
                    <span class="pill status-${slug(ticket.status)}">
                        ${esc(ticket.status)}
                    </span>
                </td>

                <td>${stamp(ticket.updated_at)}</td>
            </tr>
        `).join("")
        : `
            <tr>
                <td colspan="6" class="empty">
                    No tickets here yet. Create one to get started.
                </td>
            </tr>
        `;

    $(".table-wrap")
        .querySelectorAll("tbody tr[data-id]")
        .forEach((row) => {
            row.onclick = () => showDetail(Number(row.dataset.id));
        });

    $("#openCount").textContent = tickets.filter(
        (ticket) => ticket.status === "Open"
    ).length;

    $("#progressCount").textContent = tickets.filter(
        (ticket) => ticket.status === "In Progress"
    ).length;

    $("#resolvedCount").textContent = tickets.filter(
        (ticket) => ticket.status === "Resolved"
    ).length;
}


// ========================================
// LOAD TICKETS AND DYNAMIC FIELD DEFINITIONS
// ========================================

async function load() {
    try {
        [tickets, fields] = await Promise.all([
            request("/tickets"),
            request("/fields")
        ]);

        buildFields();
        render();
    } catch (error) {
        $("#ticketRows").innerHTML = `
            <tr>
                <td colspan="6" class="empty">
                    ${esc(error.message)}. Refresh to retry.
                </td>
            </tr>
        `;
    }
}


// ========================================
// BUILD DYNAMIC FORM FIELDS
// ========================================

function dynamicInput(field, namePrefix = "dyn_", value = "") {
    if (value == null) value = "";
    const name = `${namePrefix}${field.name}`;
    const required = field.required ? "required" : "";

    if (field.type === "dropdown") {
        return `
            <select name="${esc(name)}" ${required}>
                <option value="">Choose...</option>
                ${field.options.map((option) => `
                    <option value="${esc(option)}" ${String(option) === String(value) ? "selected" : ""}>${esc(option)}</option>
                `).join("")}
            </select>
        `;
    }

    if (field.type === "checkbox") {
        return `
            <input type="checkbox" name="${esc(name)}"
                ${value ? "checked" : ""} ${required}>
        `;
    }

    return `
        <input type="${esc(field.type)}" name="${esc(name)}"
            value="${esc(value)}" ${required}>
    `;
}

function dynamicFieldLabel(field) {
    return `${esc(field.label)}${field.required ? "" : " (optional)"}`;
}

function buildFields() {
    const container = $("#dynamicFields");

    container.innerHTML = fields.map((field) => {
        const input = dynamicInput(field);
        return `
            <label class="${field.type === "checkbox" ? "check-row" : ""}">
                ${field.type === "checkbox" ? input : ""}
                ${dynamicFieldLabel(field)}
                ${field.type === "checkbox" ? "" : input}
            </label>
        `;
    }).join("");
}


// ========================================
// CREATE TICKET FORM
// ========================================

function openForm() {
    activeTicket = null;

    $("#ticketForm").reset();
    $("#formTitle").textContent = "Create a ticket";
    $("#saveButton").textContent = "Generate response";
    $("#formError").textContent = "";

    $("#ticketDialog").showModal();
}

$("#newTicket").onclick = openForm;

$("#closeDialog").onclick = $("#cancelDialog").onclick = () => {
    $("#ticketDialog").close();
};

$("#statusFilter").onchange = render;


// ========================================
// SUBMIT NEW TICKET
// ========================================

$("#ticketForm").onsubmit = async (event) => {
    event.preventDefault();

    $("#formError").textContent = "";

    const form = event.currentTarget;
    const data = Object.fromEntries(new FormData(form));

    const dynamicValues = {};

    fields.forEach((field) => {
        const input = form.elements[`dyn_${field.name}`];

        if (input) {
            dynamicValues[field.name] = field.type === "checkbox"
                ? input.checked
                : input.value;
        }
    });

    data.dynamic_values = dynamicValues;

    try {
        const ticket = await request("/tickets", {
            method: "POST",
            body: JSON.stringify(data)
        });

        $("#ticketDialog").close();

        notify("Ticket created and response generated");

        await load();
        await showDetail(ticket.id);
    } catch (error) {
        $("#formError").textContent = error.message;
    }
};


// ========================================
// DISPLAY TICKET DETAILS
// ========================================

async function showDetail(id) {
    try {
        const ticket = await request(`/tickets/${id}`);

        activeTicket = id;

        const ticketDynamicValues = ticket.dynamic_values || {};

        $("#detailContent").innerHTML = `
            <div class="detail-wrap">

                <div class="detail-title">
                    <div>
                        <p class="eyebrow">
                            TICKET #T-${String(ticket.id).padStart(4, "0")}
                            · ${esc(ticket.status.toUpperCase())}
                        </p>

                        <h2>${esc(ticket.subject)}</h2>
                    </div>

                    <button
                        class="icon-button"
                        onclick="$('#detailDialog').close()"
                    >
                        ×
                    </button>
                </div>

                <p class="detail-sub">
                    ${esc(ticket.customer_name)}
                    · ${esc(ticket.category)}
                    · Created ${stamp(ticket.created_at)}
                </p>

                <div class="detail-section">
                    <h3>Customer description</h3>
                    <p>${esc(ticket.description)}</p>
                </div>

                ${fields.length ? `
                    <div class="detail-section">
                        <h3>Additional details</h3>
                        <div id="detailDynamicFields" class="dynamic-grid">
                            ${fields.map((field) => {
                                const input = dynamicInput(
                                    field,
                                    "detail_dyn_",
                                    ticketDynamicValues[field.name]
                                );
                                return `
                                    <label class="${field.type === "checkbox" ? "check-row" : ""}">
                                        ${field.type === "checkbox" ? input : ""}
                                        ${dynamicFieldLabel(field)}
                                        ${field.type === "checkbox" ? "" : input}
                                    </label>
                                `;
                            }).join("")}
                        </div>
                    </div>
                ` : ""}

                <div class="detail-section">
                    <h3>ML suggested solution</h3>

                    <p class="solution-box">
                        ${esc(ticket.predicted_solution)}
                    </p>
                </div>

                <div class="detail-section">
                    <h3>
                        Customer response · Review and edit before sending
                    </h3>

                    <textarea
                        class="detail-response"
                        id="editedResponse"
                    >${esc(ticket.response)}</textarea>
                </div>

                <div class="detail-actions">
                    <button class="secondary" id="deleteTicket">
                        Deactivate ticket
                    </button>

                    <div>
                        <select id="detailStatus">
                            <option ${ticket.status === "Open" ? "selected" : ""}>
                                Open
                            </option>

                            <option ${ticket.status === "In Progress" ? "selected" : ""}>
                                In Progress
                            </option>

                            <option ${ticket.status === "Resolved" ? "selected" : ""}>
                                Resolved
                            </option>
                        </select>

                        <button class="primary" id="saveResponse">
                            Save changes
                        </button>
                    </div>
                </div>

            </div>
        `;

        $("#detailDialog").showModal();

        // Save the edited response and ticket status.
        $("#saveResponse").onclick = async () => {
            try {
                await request(`/tickets/${id}`, {
                    method: "PUT",
                    body: JSON.stringify({
                        response: $("#editedResponse").value,
                        status: $("#detailStatus").value,
                        dynamic_values: Object.fromEntries(fields.flatMap((field) => {
                            const input = $(`[name="detail_dyn_${CSS.escape(field.name)}"]`);
                            if (!input) return [];
                            const value = field.type === "checkbox" ? input.checked : input.value;
                            const wasSaved = Object.prototype.hasOwnProperty.call(ticketDynamicValues, field.name);
                            if (!wasSaved && value === "") return [];
                            if (!wasSaved && field.type === "checkbox" && value === false) return [];
                            return [[field.name, value]];
                        }))
                    })
                });

                $("#detailDialog").close();

                notify("Ticket changes saved");

                await load();
            } catch (error) {
                notify(error.message);
            }
        };

        // Deactivate a ticket after confirmation.
        $("#deleteTicket").onclick = async () => {
            if (!confirm("Deactivate this ticket?")) {
                return;
            }

            try {
                await request(`/tickets/${id}`, {
                    method: "DELETE"
                });

                $("#detailDialog").close();

                notify("Ticket deactivated");

                await load();
            } catch (error) {
                notify(error.message);
            }
        };
    } catch (error) {
        notify(error.message);
    }
}


// ========================================
// ADD CUSTOM DYNAMIC FIELD
// ========================================

async function addCustomField() {
    const label = prompt("Field label?");

    if (!label || !label.trim()) {
        return;
    }

    const supportedTypes = [
        "text",
        "dropdown",
        "number",
        "date",
        "checkbox"
    ];

    const type = prompt(
        `Field type: ${supportedTypes.join(", ")}`,
        "text"
    );

    if (!supportedTypes.includes(type)) {
        notify("Choose text, dropdown, number, date, or checkbox");
        return;
    }

    let options = [];

    if (type === "dropdown") {
        options = (prompt(
            "Dropdown options, separated by commas",
            "Email, Phone"
        ) || "")
            .split(",")
            .map((option) => option.trim())
            .filter(Boolean);

        if (!options.length) {
            notify("Add at least one dropdown option");
            return;
        }
    }

    const name = label.trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_|_$/g, "");

    try {
        await request("/fields", {
            method: "POST",
            body: JSON.stringify({
                name,
                label: label.trim(),
                type,
                options,
                required: confirm("Make this field required?")
            })
        });

        fields = await request("/fields");

        buildFields();

        notify("Custom field added");
    } catch (error) {
        notify(error.message);
    }
}


// ========================================
// APPLICATION INITIALIZATION
// ========================================

// Build the existing dynamic fields and load saved tickets.
buildFields();
load();

// Add the button for creating custom fields.
const addFieldButton = document.createElement("button");

addFieldButton.type = "button";
addFieldButton.className = "secondary";
addFieldButton.textContent = "＋ Add custom field";

$("#ticketForm").insertBefore(
    addFieldButton,
    $("#ticketForm .modal-actions")
);

addFieldButton.onclick = addCustomField;
