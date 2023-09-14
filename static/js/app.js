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
  
    onSendButton(chatbox) {
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        let textField = chatbox.querySelector('input');
        console.log('Button clicked'); 
        var text1 = textField.value;
        
        // Assurez-vous que le champ n'est pas vide avant d'envoyer la requête POST
        if (text1 === "") {
            return;
        }
        console.log("sending request")
        fetch('/send_gesture/')
        .then(
            response => response.json())
        .then(data => {  
            if (data.hasOwnProperty("gesture")) {      
            const gesture = data.gesture
            textField.value = gesture
            text1 = textField.value;
            console.log("gesture", text1)
            
            
            
        }
            else console.log("pas de gesture")
          
    }).catch((error) => {
        console.error('Error: ', error);
        });

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);
    
      
    
        fetch('/chatbot/', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken 
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';
        })
        .catch((error) => {
            console.error(error);
            this.updateChatText(chatbox);
            textField.value = '';
        });
    }
    
    
    // updateChatText(chatbox) {
    //     const chatMessagesDiv = chatbox.querySelector('.chatbox__messages');
    //     chatMessagesDiv.innerHTML = ''; // Clear existing messages
      
    //     this.messages.slice().reverse().forEach((item) => {
    //         const messageDiv = document.createElement('div');
    //         messageDiv.classList.add('messages__item');
    //         if (item.name === 'Sam') {
    //             messageDiv.classList.add('messages__item--operator');
    //         } else {
    //             messageDiv.classList.add('messages__item--visitor');
    //         }
    
    //         if (item.message.startsWith('http') && item.message.endsWith('.gif')) {
    //             const gifImg = document.createElement('img');
    //             gifImg.classList.add('gif-message');
    //             gifImg.src = item.message;
    //             messageDiv.appendChild(gifImg);
    //         } else {
    //             messageDiv.textContent = item.message;
    //         }
    
    //         chatMessagesDiv.appendChild(messageDiv);
    //     });
      
    //     // Scroll to the bottom to show the latest message
    //     chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
    // }
    updateChatText(chatbox) {
        const chatMessagesDiv = chatbox.querySelector('.chatbox__messages');
        chatMessagesDiv.innerHTML = ''; // Clear existing messages
    
        this.messages.slice().reverse().forEach((item) => {
            if (item.message.startsWith('http')) {
                // If the message is a GIF URL, create an <img> element
                const gifElement = document.createElement('img');
                gifElement.classList.add('gif-message');
                gifElement.src = item.message;
                gifElement.alt = 'GIF';
                chatMessagesDiv.appendChild(gifElement); // Append <img> to messages container
            } else {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('messages__item');
    
                if (item.name === 'Sam') {
                    messageDiv.classList.add('messages__item--operator');
                } else {
                    messageDiv.classList.add('messages__item--visitor');
                }
    
                messageDiv.textContent = item.message;
                chatMessagesDiv.appendChild(messageDiv);
            }
        });
    
        // Scroll to the bottom to show the latest message
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight;
    }
    
    
    


}

const chatbox = new Chatbox();
chatbox.display();