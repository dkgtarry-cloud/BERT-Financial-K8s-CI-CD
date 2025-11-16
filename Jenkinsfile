pipeline {
    agent any

    environment {
        HARBOR_URL = "192.168.0.137:8081"
        HARBOR_PROJECT = "financial-nlp"
        IMAGE_NAME = "financial-nlp-api"
        IMAGE_TAG = "v1"
    }

    stages {
        stage('Build Image') {
            steps {
                sh """
                    docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG} .
                """
            }
        }

        stage('Push Image to Harbor') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'harbor-cred', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
                    sh """
                        echo $PASS | docker login ${HARBOR_URL} -u $USER --password-stdin
                        docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}
                    """
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                withCredentials([file(credentialsId: 'k8s-config', variable: 'KUBECONFIG')]) {
                    sh """
                        kubectl --kubeconfig=$KUBECONFIG rollout restart deployment/financial-nlp-deploy
                    """
                }
            }
        }
    }
}
