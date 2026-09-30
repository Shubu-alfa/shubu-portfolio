class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        };

        this.state = false;
        this.messages = [];
        this.sessionId = this.getOrCreateSessionId();
    }

    getOrCreateSessionId() {
        const key = 'shubzfolio_chat_session_id';
        let id = sessionStorage.getItem(key);
        if (!id) {
            if (window.crypto && crypto.randomUUID) {
                id = crypto.randomUUID();
            } else {
                id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
                    const r = Math.random() * 16 | 0;
                    const v = c === 'x' ? r : (r & 0x3 | 0x8);
                    return v.toString(16);
                });
            }
            sessionStorage.setItem(key, id);
        }
        return id;
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;
        if (!openButton || !chatBox || !sendButton) {
            return;
        }

        openButton.addEventListener('click', () => this.toggleState(chatBox, openButton));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        if (node) {
            node.addEventListener('keyup', ({ key }) => {
                if (key === 'Enter') {
                    this.onSendButton(chatBox);
                }
            });
        }
    }

    toggleState(chatbox, openButton) {
        this.state = !this.state;
        if (this.state) {
            chatbox.classList.add('chatbox--active');
            if (openButton) {
                openButton.setAttribute('aria-label', 'Close chat');
            }
        } else {
            chatbox.classList.remove('chatbox--active');
            if (openButton) {
                openButton.setAttribute('aria-label', 'Open chat');
            }
        }
    }

    escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    onSendButton(chatbox) {
        const textField = chatbox.querySelector('input');
        if (!textField) {
            return;
        }
        const text1 = textField.value.trim();
        if (text1 === '') {
            return;
        }

        this.messages.push({ name: 'User', message: text1 });
        textField.value = '';
        this.updateChatText(chatbox);

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1, session_id: this.sessionId }),
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then((r) => r.json())
            .then((r) => {
                if (r.session_id) {
                    this.sessionId = r.session_id;
                    sessionStorage.setItem('shubzfolio_chat_session_id', r.session_id);
                }
                this.messages.push({ name: 'Shubh', message: r.answer || 'No response.' });
                this.updateChatText(chatbox);
            })
            .catch((error) => {
                console.error('Error:', error);
                this.messages.push({ name: 'Shubh', message: 'Sorry, something went wrong. Please try again.' });
                this.updateChatText(chatbox);
            });
    }

    updateChatText(chatbox) {
        let html = '';
        this.messages.slice().reverse().forEach((item) => {
            const safe = this.escapeHtml(item.message);
            if (item.name === 'Shubh') {
                html += '<div class="messages__item messages__item--visitor">' + safe + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + safe + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        if (chatmessage) {
            chatmessage.innerHTML = html;
        }
    }
}

document.addEventListener('DOMContentLoaded', function () {
    const chatbox = new Chatbox();
    chatbox.display();
});
