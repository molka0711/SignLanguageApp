class Chatbox{
    constructor() {
        this.args = {
        openButton: document.querySelector('.chatbox__button'),
        chatBox: document.querySelector('.chatbox__support'),
        sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
        this.updateChatText = this.updateChatText.bind(this); // Bind the method to the instance
    }

    display(){
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox) )
        sendButton.addEventListener('click', () => this.onSendButton(chatBox) )
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter"){
            this.onSendButton(chatBox)
            }

        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        //show or hide the box
        if(this.state){
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox){
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

        fetch( '/chatbot/',
            {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            // mode: 'cors',
            mode: 'same-origin',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken 
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
            });
    }

    updateChatText(chatbox) {
      const chatMessagesDiv = chatbox.querySelector('.chatbox__messages');
      chatMessagesDiv.innerHTML = ''; // Clear existing messages

      this.messages.slice().reverse().forEach(function (item) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('messages__item');
        if (item.name === 'Sam') {
          messageDiv.classList.add('messages__item--operator');
        } else {
          messageDiv.classList.add('messages__item--visitor');
        }
        messageDiv.textContent = item.message;
        chatMessagesDiv.appendChild(messageDiv);
      });

      // Scroll to the bottom to show the latest message
      chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
    }



}

const chatbox = new Chatbox();
chatbox.display();